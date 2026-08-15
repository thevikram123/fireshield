from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .calibrate import calibrate_auto, parse_overall
from .commercial_model import build_commercial_model
from .correction import constrain_geometry_to_spec
from .export_dxf import write_dxf
from .geometry import extract_geometry
from .ingest import load_raster
from .ocr import run_ocr
from .overlay import write_overlay
from .preprocess import preprocess
from .recognize import MissingWeights, Recognition, load_model, recognize, remap_recognition
from .reconstruct import apply_scale, reconstruct
from .schema import FloorplanModel
from .topology import derive_topology


@dataclass
class ConvertResult:
    model: FloorplanModel
    dxf_path: Path
    json_path: Optional[Path]
    overlay_path: Optional[Path]
    mm_per_px: Optional[float]
    warnings: list[str]
    commercial_model: dict


def convert(
    path: str | Path,
    *,
    out: str | Path | None = None,
    page: int = 0,
    dpi: int = 200,
    weights: str | Path | None = None,
    device: Optional[str] = None,
    tta: bool = False,
    ocr: bool = True,
    mm_per_px: Optional[float] = None,
    overall: Optional[str] = None,
    max_side: int = 1024,
    threshold: float = 0.2,
    overlay: str | Path | bool = True,
    json_out: str | Path | bool = True,
    model=None,
    use_model: bool = True,
    guide: Optional[Callable] = None,
    guide_required: bool = False,
) -> ConvertResult:
    warnings: list[str] = []
    src = Path(path)
    raster = load_raster(src, page=page, dpi=dpi)
    prep = preprocess(raster.rgb, max_side=max_side)
    plan_spec = {}
    if guide is not None and hasattr(guide, "specify"):
        try:
            plan_spec = guide.specify(prep.display_rgb)
        except Exception as exc:
            if guide_required:
                raise
            warnings.append(f"Vision-first specification failed ({exc}); using unconstrained extraction.")
    cv_geom = extract_geometry(prep.display_rgb, prep.wall_mask)
    if plan_spec:
        constraint = constrain_geometry_to_spec(
            cv_geom, plan_spec, (prep.display_rgb.shape[1], prep.display_rgb.shape[0]),
            rgb=prep.display_rgb,
        )
        if constraint.get("applied"):
            warnings.append(
                "Vision-first constraints applied: "
                f"{constraint['matchedWalls']}/{constraint['specifiedWalls']} walls and "
                f"{constraint['matchedRooms']}/{constraint['specifiedRooms']} rooms matched to pixels."
            )

    recog = Recognition(
        polygons=None,
        types=[],
        room_polygons=[],
        room_types=[],
        heatmaps=None,
        rooms=None,
        icons=None,
    )
    if use_model:
        try:
            if model is None:
                net, device = load_model(Path(weights) if weights else None, device=device)
            else:
                net = model
                if device is None:
                    device = "cpu"
            recog = recognize(prep, net, device=device, tta=tta, threshold=threshold)
            recog = remap_recognition(recog, prep)
        except MissingWeights as exc:
            warnings.append(str(exc))
        except Exception as exc:
            warnings.append(f"CubiCasa recognition failed ({exc}); using OpenCV walls/rooms.")

    try:
        texts = run_ocr(prep.display_rgb, enabled=ocr)
    except RuntimeError as exc:
        warnings.append(str(exc))
        texts = []

    plan = reconstruct(
        recog.polygons,
        recog.types,
        recog.room_polygons,
        recog.room_types,
        texts,
        image_size=(prep.display_rgb.shape[1], prep.display_rgb.shape[0]),
        cv_geom=cv_geom,
    )

    if guide is not None:
        try:
            plan = guide(prep.display_rgb, plan)
        except Exception as exc:
            if guide_required:
                raise
            warnings.append(f"Vision guidance failed ({exc}); keeping deterministic topology.")

    if cv_geom.envelope:
        x0, y0, x1, y1 = cv_geom.envelope
        plan.exterior_boundary = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    plan = derive_topology(plan)

    overall_mm = parse_overall(overall) if overall else None
    bar_h = next((b.length_px for b in cv_geom.bars if b.horizontal), None)
    bar_v = next((b.length_px for b in cv_geom.bars if not b.horizontal), None)
    scale = calibrate_auto(
        plan.rooms,
        plan.texts,
        override=mm_per_px,
        overall_mm=overall_mm,
        envelope=cv_geom.envelope,
        bar_h_px=bar_h,
        bar_v_px=bar_v,
    )
    if scale is None:
        warnings.append(
            "Could not infer scale from printed dimensions. DXF stays in pixel units. "
            "Pass --overall W,H (any units the drawing uses) or --mm-per-px."
        )

    stem = src.stem if src.stem else "floorplan"
    out_path = Path(out) if out else Path("out") / f"{stem}.dxf"
    overlay_path = None
    if overlay:
        overlay_path = (
            Path(overlay) if isinstance(overlay, (str, Path)) else out_path.with_name(out_path.stem + "_overlay.png")
        )
        write_overlay(prep.display_rgb, plan, overlay_path)

    plan = apply_scale(plan, scale)
    dxf_path = write_dxf(plan, out_path)

    json_path = None
    if json_out:
        json_path = Path(json_out) if isinstance(json_out, (str, Path)) else dxf_path.with_suffix(".json")
        commercial_model = build_commercial_model(
            plan,
            building_profile=getattr(guide, "building_profile", {}),
            plan_spec=plan_spec,
            guidance_audit=vars(guide.audit) if guide is not None and hasattr(guide, "audit") else {},
        )
        json_path.write_text(json.dumps({
            "topology": plan.to_dict(), "commercialModel": commercial_model,
        }, indent=2), encoding="utf-8")
    else:
        commercial_model = build_commercial_model(
            plan,
            building_profile=getattr(guide, "building_profile", {}),
            plan_spec=plan_spec,
            guidance_audit=vars(guide.audit) if guide is not None and hasattr(guide, "audit") else {},
        )

    return ConvertResult(
        model=plan,
        dxf_path=dxf_path,
        json_path=json_path,
        overlay_path=overlay_path,
        mm_per_px=scale,
        warnings=warnings,
        commercial_model=commercial_model,
    )
