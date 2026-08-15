"""Bounded Qwen-vision corrections for deterministic floor-plan topology.

The vision model may propose additions, but this module validates every proposal
against the source image coordinate system. It never lets an LLM emit DXF
entities directly.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from .schema import Door, FloorplanModel, Wall, Window


@dataclass
class GuidanceAudit:
    review_status: str = "not_run"
    review_confidence: float = 0.0
    review_summary: str = ""
    discrepancies: list[dict] = field(default_factory=list)
    accepted: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)


class QwenTopologyGuide:
    """Ask Qwen to identify omissions, then apply only safe geometric additions."""

    def __init__(self, api_key: str | None = None, model: str = "qwen/qwen3.6-27b"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.audit = GuidanceAudit()
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")

    def __call__(self, rgb: np.ndarray, model: FloorplanModel) -> FloorplanModel:
        payload = self._request(rgb, model)
        review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
        status = str(review.get("status", "insufficient"))
        self.audit.review_status = status if status in {"approved", "needs_correction", "insufficient"} else "insufficient"
        self.audit.review_confidence = max(0.0, min(float(review.get("confidence", 0)), 1.0))
        self.audit.review_summary = str(review.get("summary", ""))[:500]
        self.audit.discrepancies = list(review.get("discrepancies") or [])[:50]
        return apply_guidance(model, payload, self.audit)

    def _request(self, rgb: np.ndarray, model: FloorplanModel) -> dict:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("could not encode plan image")
        image_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        compact = {
            "image_size": list(model.image_size),
            "walls": [{"id": w.id, "start": w.start, "end": w.end} for w in model.walls],
            "doors": [{"center": d.center, "width": d.width_mm} for d in model.doors],
            "windows": [{"center": w.center, "width": w.width_mm} for w in model.windows],
            "rooms": [{"id": r.id, "type": r.type, "boundary": r.boundary} for r in model.rooms],
        }
        prompt = (
            "Compare this floor-plan image with the extracted topology JSON. Propose only clearly visible "
            "objects that the extraction missed. Coordinates must be pixels in the supplied image_size. "
            "Walls must be horizontal or vertical centerlines. Do not delete or move existing geometry. "
            "First review whether the extracted structure matches the visible drawing. "
            "Return JSON only: {\"review\":{\"status\":\"approved|needs_correction|insufficient\","
            "\"confidence\":0..1,\"summary\":str,\"discrepancies\":[{\"kind\":str,\"description\":str}]},"
            "\"add_walls\":[{\"start\":[x,y],\"end\":[x,y],\"thickness\":px,"
            "\"confidence\":0..1}],\"add_doors\":[{\"center\":[x,y],\"width\":px,\"confidence\":0..1}],"
            "\"add_windows\":[same],\"room_labels\":[{\"room_id\":str,\"label\":str,\"type\":str,"
            "\"confidence\":0..1}]}. Omit uncertain proposals. Extracted topology: " + json.dumps(compact)
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            "temperature": 0.1,
            "max_completion_tokens": 1800,
            "response_format": {"type": "json_object"},
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Qwen request failed ({exc.code})") from exc
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return json.loads(content)


def apply_guidance(
    model: FloorplanModel,
    proposals: dict,
    audit: GuidanceAudit | None = None,
    min_confidence: float = 0.75,
) -> FloorplanModel:
    audit = audit or GuidanceAudit()
    width, height = model.image_size

    def point(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        x, y = float(value[0]), float(value[1])
        return (x, y) if 0 <= x <= width and 0 <= y <= height else None

    for raw in list(proposals.get("add_walls") or [])[:64]:
        try:
            a, b = point(raw.get("start")), point(raw.get("end"))
            conf = float(raw.get("confidence", 0))
            thick = max(2.0, min(float(raw.get("thickness", 8)), 80.0))
            if not a or not b or conf < min_confidence:
                raise ValueError("low confidence or invalid coordinates")
            if abs(a[0] - b[0]) <= 3:
                x = (a[0] + b[0]) / 2
                a, b = (x, a[1]), (x, b[1])
            elif abs(a[1] - b[1]) <= 3:
                y = (a[1] + b[1]) / 2
                a, b = (a[0], y), (b[0], y)
            else:
                raise ValueError("wall is not axis-aligned")
            if np.hypot(b[0] - a[0], b[1] - a[1]) < 8:
                raise ValueError("wall is too short")
            if any(_same_segment(a, b, w.start, w.end, 6) for w in model.walls):
                raise ValueError("duplicate wall")
            item = Wall(id=f"wall_{len(model.walls)}", start=a, end=b, thickness_mm=thick)
            model.walls.append(item)
            audit.accepted.append({"kind": "wall", "id": item.id, "confidence": conf})
        except (TypeError, ValueError) as exc:
            audit.rejected.append({"kind": "wall", "proposal": raw, "reason": str(exc)})

    for key, cls, target in (("add_doors", Door, model.doors), ("add_windows", Window, model.windows)):
        for raw in list(proposals.get(key) or [])[:32]:
            try:
                center = point(raw.get("center"))
                conf = float(raw.get("confidence", 0))
                size = float(raw.get("width", 0))
                if not center or conf < min_confidence or not 4 <= size <= max(width, height) / 2:
                    raise ValueError("low confidence or invalid opening")
                common = {"id": f"{'door' if cls is Door else 'window'}_{len(target)}", "wall_id": None,
                          "width_mm": size, "center": center}
                item = cls(**common)
                target.append(item)
                audit.accepted.append({"kind": "door" if cls is Door else "window", "id": item.id,
                                       "confidence": conf})
            except (TypeError, ValueError) as exc:
                audit.rejected.append({"kind": key, "proposal": raw, "reason": str(exc)})

    rooms = {room.id: room for room in model.rooms}
    for raw in list(proposals.get("room_labels") or [])[:64]:
        room = rooms.get(str(raw.get("room_id", "")))
        conf = float(raw.get("confidence", 0))
        if room and conf >= min_confidence:
            room.label = str(raw.get("label", ""))[:80] or room.label
            room.type = str(raw.get("type", room.type))[:40].upper()
            audit.accepted.append({"kind": "room_label", "id": room.id, "confidence": conf})
        else:
            audit.rejected.append({"kind": "room_label", "proposal": raw, "reason": "unknown room or low confidence"})
    return model


def _same_segment(a, b, c, d, tolerance: float) -> bool:
    direct = np.hypot(a[0] - c[0], a[1] - c[1]) + np.hypot(b[0] - d[0], b[1] - d[1])
    reverse = np.hypot(a[0] - d[0], a[1] - d[1]) + np.hypot(b[0] - c[0], b[1] - c[1])
    return min(direct, reverse) <= tolerance * 2
