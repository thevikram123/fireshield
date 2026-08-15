from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Preprocessed:
    model_rgb: np.ndarray
    display_rgb: np.ndarray
    wall_mask: np.ndarray
    scale: float
    pad: tuple[int, int, int, int]
    original_size: tuple[int, int]
    angle_deg: float


def preprocess(rgb: np.ndarray, max_side: int = 1024) -> Preprocessed:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image")
    orig_h, orig_w = rgb.shape[:2]
    deskewed, angle = deskew(rgb)
    cleaned = denoise(deskewed)
    wall_mask = isolate_wall_strokes(cleaned)
    model_rgb, scale, pad = letterbox(cleaned, max_side=max_side, multiple=64)
    return Preprocessed(
        model_rgb=model_rgb,
        display_rgb=deskewed,
        wall_mask=wall_mask,
        scale=scale,
        pad=pad,
        original_size=(orig_w, orig_h),
        angle_deg=angle,
    )


def deskew(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        return rgb, 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = angle + 90
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.5:
        return rgb, 0.0
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return rotated, float(angle)


def denoise(rgb: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(rgb, d=5, sigmaColor=40, sigmaSpace=40)


def isolate_wall_strokes(rgb: np.ndarray) -> np.ndarray:
    """Keep thick architectural ink; drop furniture, text, and colored dimension bars."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary[hsv[:, :, 1] > 70] = 0
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=1)
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 3)
    thick = (dist >= 1.4).astype(np.uint8) * 255
    if cv2.countNonZero(thick) < 80:
        thick = opened
    restore = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(thick, restore, iterations=1)


def letterbox(rgb: np.ndarray, max_side: int = 1024, multiple: int = 64) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
    h, w = rgb.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas_w = int(np.ceil(new_w / multiple) * multiple)
    canvas_h = int(np.ceil(new_h / multiple) * multiple)
    pad_left = (canvas_w - new_w) // 2
    pad_right = canvas_w - new_w - pad_left
    pad_top = (canvas_h - new_h) // 2
    pad_bottom = canvas_h - new_h - pad_top
    canvas = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )
    return canvas, float(scale), (pad_left, pad_top, pad_right, pad_bottom)


def map_xy_to_display(x: float, y: float, prep: Preprocessed) -> tuple[float, float]:
    pad_l, pad_t, _, _ = prep.pad
    return ((x - pad_l) / prep.scale, (y - pad_t) / prep.scale)
