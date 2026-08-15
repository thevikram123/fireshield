"""HTTP boundary for FireShield plan conversion.

Designed to run behind the authenticated/rate-limited Cloudflare Worker. It has
no browser CORS policy and no embedded provider credentials.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from .guidance import QwenTopologyGuide
from .pipeline import convert


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_TYPES = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

app = FastAPI(title="FireShield Floor Plan Processor", docs_url=None, redoc_url=None)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "fireshield-floorplan",
        "qwenConfigured": bool(os.environ.get("GROQ_API_KEY")),
        "requestScopedQwenSupported": True,
        "serviceTokenConfigured": bool(os.environ.get("FLOORPLAN_SERVICE_TOKEN")),
        "cubicCasaEnabled": os.environ.get("ENABLE_NONCOMMERCIAL_CUBICASA") == "1",
    }


@app.post("/convert")
async def convert_plan(
    file: UploadFile = File(...),
    overall: str | None = Form(default=None),
    page: int = Form(default=0),
    require_qwen: bool = Form(default=True),
    building_profile: str = Form(default="{}"),
    service_token: str | None = Header(default=None, alias="X-FireShield-Service-Token"),
    groq_api_key: str | None = Header(default=None, alias="X-FireShield-Groq-Key"),
):
    _require_service_token(service_token)
    mime = (file.content_type or "").lower()
    suffix = ALLOWED_TYPES.get(mime)
    if suffix is None:
        raise HTTPException(415, "Upload a PDF, PNG, JPEG or WebP floor plan.")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(400, "The uploaded plan is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The floor plan exceeds the 25 MB limit.")
    if page < 0 or page > 100:
        raise HTTPException(400, "PDF page must be between 0 and 100.")
    profile = _building_profile(building_profile)

    guide = None
    if require_qwen:
        try:
            guide = QwenTopologyGuide(api_key=groq_api_key)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    try:
        with tempfile.TemporaryDirectory(prefix="fireshield-plan-") as temp:
            work = Path(temp)
            source = work / f"source{suffix}"
            source.write_bytes(payload)
            dxf = work / "plan.dxf"
            result = convert(
                source,
                out=dxf,
                page=page,
                overall=overall or None,
                weights=Path("/app/weights/model_best_val_loss_var.pkl"),
                use_model=os.environ.get("ENABLE_NONCOMMERCIAL_CUBICASA") == "1",
                guide=guide,
                guide_required=require_qwen,
            )
            json_path = result.json_path
            overlay_path = result.overlay_path
            return {
                "buildingProfile": profile,
                "topology": result.model.to_dict(),
                "warnings": result.warnings,
                "guidance": {
                    "required": require_qwen,
                    "reviewStatus": guide.audit.review_status if guide else "not_run",
                    "reviewConfidence": guide.audit.review_confidence if guide else 0,
                    "reviewSummary": guide.audit.review_summary if guide else "",
                    "discrepancies": guide.audit.discrepancies if guide else [],
                    "accepted": guide.audit.accepted if guide else [],
                    "rejected": guide.audit.rejected if guide else [],
                },
                "artifacts": {
                    "plan.dxf": _artifact(dxf, "application/dxf"),
                    "plan.json": _artifact(json_path, "application/json") if json_path else None,
                    "plan_overlay.png": _artifact(overlay_path, "image/png") if overlay_path else None,
                },
                "metrics": {
                    "walls": len(result.model.walls),
                    "doors": len(result.model.doors),
                    "windows": len(result.model.windows),
                    "rooms": len(result.model.rooms),
                    "connections": len(result.model.connections),
                    "objects": len(result.model.objects),
                    "texts": len(result.model.texts),
                    "units": result.model.units,
                    "mmPerPx": result.mm_per_px,
                },
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Plan conversion failed: {exc}") from exc


def _artifact(path: Path, mime: str) -> dict:
    data = path.read_bytes()
    return {
        "mimeType": mime,
        "byteSize": len(data),
        "base64": base64.b64encode(data).decode("ascii"),
    }


def _building_profile(raw: str) -> dict:
    if len(raw) > 4000:
        raise HTTPException(400, "Building profile is too large.")
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Building profile must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise HTTPException(400, "Building profile must be a JSON object.")
    allowed = {
        "building", "level", "occupancy", "buildingHeightM", "floorAreaM2",
        "expectedOccupants", "notes",
    }
    return {
        key: (str(item)[:500] if isinstance(item, str) else item)
        for key, item in value.items()
        if key in allowed and isinstance(item, (str, int, float, bool))
    }


def _require_service_token(received: str | None) -> None:
    expected = os.environ.get("FLOORPLAN_SERVICE_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Floor-plan service authentication is not configured.")
    if not received or not hmac.compare_digest(received, expected):
        raise HTTPException(401, "Invalid floor-plan service credentials.")
