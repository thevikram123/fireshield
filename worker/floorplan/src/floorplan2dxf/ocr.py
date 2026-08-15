from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import numpy as np

from .calibrate import classify_text, parse_dimension_pair
from .schema import LABEL_TO_ROOM_TYPE, TextItem


def run_ocr(rgb: np.ndarray, enabled: bool = True) -> list[TextItem]:
    if not enabled:
        return []
    reader = _reader()
    items: list[TextItem] = []
    for box, text, conf in reader.readtext(rgb):
        content = (text or "").strip()
        if not content:
            continue
        pts = [(float(p[0]), float(p[1])) for p in box]
        cx = sum(p[0] for p in pts) / max(len(pts), 1)
        cy = sum(p[1] for p in pts) / max(len(pts), 1)
        kind = classify_text(content)
        parsed = parse_dimension_pair(content) if kind == "dimension" else None
        items.append(
            TextItem(
                content=content,
                position=(cx, cy),
                kind=kind,  # type: ignore[arg-type]
                bbox=pts,
                confidence=float(conf),
                parsed_mm=parsed,
            )
        )
    return items


@lru_cache(maxsize=1)
def _reader():
    import easyocr

    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False,
        model_storage_directory=os.environ.get("EASYOCR_MODEL_DIR"),
    )


def label_to_room_type(text: str) -> Optional[str]:
    key = " ".join(text.lower().split())
    if key in LABEL_TO_ROOM_TYPE:
        return LABEL_TO_ROOM_TYPE[key]
    for token in key.replace("/", " ").split():
        if token in LABEL_TO_ROOM_TYPE:
            return LABEL_TO_ROOM_TYPE[token]
    return None
