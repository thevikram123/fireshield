from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="floorplan2dxf",
        description="PDF/image → CubiCasa → OCR → topology → scaled DXF",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download-weights", help="Download CubiCasa checkpoint")
    p_dl.add_argument("--dest", type=Path, default=None)

    p_cv = sub.add_parser("convert", help="Convert a floor-plan scan to DXF")
    p_cv.add_argument("input", type=Path)
    p_cv.add_argument("--out", type=Path, default=None)
    p_cv.add_argument("--page", type=int, default=0)
    p_cv.add_argument("--dpi", type=int, default=200)
    p_cv.add_argument("--weights", type=Path, default=None)
    p_cv.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p_cv.add_argument("--tta", action="store_true")
    p_cv.add_argument("--no-ocr", action="store_true")
    p_cv.add_argument(
        "--overall",
        default=None,
        help="Printed overall size from THIS drawing, e.g. 40ft,30ft or 12000x9000mm",
    )
    p_cv.add_argument("--mm-per-px", type=float, default=None)
    p_cv.add_argument("--no-model", action="store_true", help="Skip CubiCasa; OpenCV walls/rooms only")
    p_cv.add_argument("--max-side", type=int, default=1024)
    p_cv.add_argument("--threshold", type=float, default=0.2)
    p_cv.add_argument("--overlay", type=Path, default=None)
    p_cv.add_argument("--json", dest="json_out", type=Path, default=None)
    p_cv.add_argument("--no-overlay", action="store_true")
    p_cv.add_argument("--no-json", action="store_true")

    p_demo = sub.add_parser("demo", help="Write the semantic bedroom example DXF/JSON (no model needed)")
    p_demo.add_argument("--out-dir", type=Path, default=Path("out"))

    args = parser.parse_args(argv)
    if args.cmd == "download-weights":
        from .recognize import download_weights

        dest = download_weights(args.dest)
        print(f"Weights ready: {dest}")
        return 0

    if args.cmd == "demo":
        from .demo import write_demo

        dxf, payload = write_demo(args.out_dir)
        print(f"DXF   {dxf}")
        print(f"JSON  {payload}")
        return 0

    from .pipeline import convert

    overlay: Path | bool
    if args.no_overlay:
        overlay = False
    elif args.overlay:
        overlay = args.overlay
    else:
        overlay = True

    json_out: Path | bool
    if args.no_json:
        json_out = False
    elif args.json_out:
        json_out = args.json_out
    else:
        json_out = True

    result = convert(
        args.input,
        out=args.out,
        page=args.page,
        dpi=args.dpi,
        weights=args.weights,
        device=args.device,
        tta=args.tta,
        ocr=not args.no_ocr,
        mm_per_px=args.mm_per_px,
        overall=args.overall,
        use_model=not args.no_model,
        max_side=args.max_side,
        threshold=args.threshold,
        overlay=overlay,
        json_out=json_out,
    )
    print(f"DXF      {result.dxf_path}")
    if result.json_path:
        print(f"JSON     {result.json_path}")
    if result.overlay_path:
        print(f"overlay  {result.overlay_path}")
    print(f"units    {result.model.units}")
    print(f"mm/px    {result.mm_per_px}")
    print(
        f"objects  walls={len(result.model.walls)} doors={len(result.model.doors)} "
        f"windows={len(result.model.windows)} rooms={len(result.model.rooms)} "
        f"texts={len(result.model.texts)}"
    )
    for room in result.model.rooms:
        extra = ""
        if room.label:
            extra += f" label={room.label!r}"
        if room.dimension_text:
            extra += f" dim={room.dimension_text!r}"
        print(f"  ROOM {room.id} type={room.type}{extra}")
    for warning in result.warnings:
        print(f"warning  {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
