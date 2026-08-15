"""Deterministic door/window detection from architectural drawing convention.

Standard plan symbols, which this encodes directly:

* A **door** is a quarter-circle swing arc plus a straight leaf. The arc bulges
  *perpendicular, away from* the wall into the room it opens onto.
* A **window** is a thin box drawn *inside* the wall band — parallel thin lines
  collinear with the wall, never bulging out of it. A wide window is drawn as
  several collinear blocks, so adjacent blocks on the same wall are merged into
  one opening.

That perpendicular-bulge-vs-collinear distinction is what separates the two, and
it is a property of the convention rather than of any particular drawing: every
threshold here is a ratio of the image's own dimensions or of the local wall
thickness, so nothing is tuned to a specific plan.

Unlike a vision model's read, the widths produced here are real pixel geometry,
so once a scale is recovered they become actual clear-width figures that can be
checked against a regulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .schema import Point, Wall


@dataclass
class Opening:
    kind: str  # "door" | "window"
    center: Point
    width_px: float
    wall_id: str | None
    horizontal: bool
    confidence: float


def detect_openings(
    rgb: np.ndarray,
    wall_mask: np.ndarray,
    walls: list[Wall],
) -> list[Opening]:
    if rgb is None or rgb.size == 0 or not walls:
        return []
    height, width = rgb.shape[:2]
    scale = min(height, width)
    # Reference sizes derived from the drawing itself, not from any one plan.
    door_span = max(12.0, scale * 0.045)      # a single-leaf door is ~4.5% of the short side
    stroke = max(2, int(round(scale * 0.004)))  # nominal thin-line stroke

    thin = _thin_ink(rgb, stroke)
    if thin is None or not thin.any():
        return []

    windows = _detect_windows(thin, walls, door_span, stroke)
    doors = _detect_doors(thin, walls, door_span, windows)
    return _dedupe(doors + windows, min_separation=door_span * 0.5)


def _thin_ink(rgb: np.ndarray, stroke: int) -> np.ndarray:
    """Ink with the solid wall bands removed, leaving door leaves/arcs and window boxes."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # A wall band survives an opening with a kernel wider than a thin stroke;
    # arcs, leaves and window boxes do not.
    band = max(3, stroke * 3)
    solid = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (band, band)))
    return cv2.subtract(ink, solid)


def _wall_axis(wall: Wall) -> tuple[bool, float, float, float]:
    """Return (horizontal, fixed-axis coordinate, span start, span end)."""
    (x1, y1), (x2, y2) = wall.start, wall.end
    if abs(x2 - x1) >= abs(y2 - y1):
        return True, (y1 + y2) / 2.0, min(x1, x2), max(x1, x2)
    return False, (x1 + x2) / 2.0, min(y1, y2), max(y1, y2)


def _nearest_wall(
    cx: float, cy: float, walls: list[Wall], tolerance: float,
) -> tuple[Wall | None, bool, float]:
    """Nearest wall whose band contains this point, with its orientation."""
    best: tuple[Wall | None, bool, float] = (None, True, float("inf"))
    for wall in walls:
        horizontal, fixed, lo, hi = _wall_axis(wall)
        along, across = (cx, cy) if horizontal else (cy, cx)
        if along < lo - tolerance or along > hi + tolerance:
            continue
        distance = abs(across - fixed)
        if distance < best[2]:
            best = (wall, horizontal, distance)
    if best[2] > tolerance:
        return (None, True, best[2])
    return best


def _detect_windows(
    thin: np.ndarray, walls: list[Wall], door_span: float, stroke: int,
) -> list[Opening]:
    """Thin boxes lying inside a wall band, with collinear blocks merged."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(thin, connectivity=8)
    # Tolerance for "inside the wall band" — a window box sits within the wall,
    # so allow roughly a wall thickness either side.
    band_tolerance = max(4.0, door_span * 0.35)
    blocks: list[tuple[Wall, bool, float, float, float]] = []  # wall, horiz, along, across, length
    for index in range(1, count):
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < max(6, stroke * 3):
            continue
        long_side, short_side = (max(w, h), min(w, h))
        if long_side < max(5.0, door_span * 0.18):
            continue
        # A window block is elongated and thin — a blob or a big shape is not one.
        if short_side > max(4.0, door_span * 0.5) or long_side < short_side * 1.6:
            continue
        cx, cy = float(centroids[index][0]), float(centroids[index][1])
        wall, horizontal, _ = _nearest_wall(cx, cy, walls, band_tolerance)
        if wall is None:
            continue
        # Its long axis must run ALONG the wall (a door leaf crosses it instead).
        block_horizontal = w >= h
        if block_horizontal != horizontal:
            continue
        along, across = (cx, cy) if horizontal else (cy, cx)
        blocks.append((wall, horizontal, along, across, float(long_side)))

    return _merge_collinear(blocks, gap=max(6.0, door_span * 0.9))


def _merge_collinear(
    blocks: list[tuple[Wall, bool, float, float, float]], gap: float,
) -> list[Opening]:
    """Merge adjacent blocks on the same wall — a wide window is drawn as several."""
    grouped: dict[str, list[tuple[Wall, bool, float, float, float]]] = {}
    for block in blocks:
        grouped.setdefault(block[0].id, []).append(block)

    openings: list[Opening] = []
    for items in grouped.values():
        items.sort(key=lambda item: item[2])
        run = [items[0]]
        for block in items[1:]:
            previous = run[-1]
            previous_end = previous[2] + previous[4] / 2.0
            block_start = block[2] - block[4] / 2.0
            if block_start - previous_end <= gap:
                run.append(block)
            else:
                openings.append(_opening_from_run(run))
                run = [block]
        openings.append(_opening_from_run(run))
    return openings


def _opening_from_run(run: list[tuple[Wall, bool, float, float, float]]) -> Opening:
    wall, horizontal = run[0][0], run[0][1]
    start = min(block[2] - block[4] / 2.0 for block in run)
    end = max(block[2] + block[4] / 2.0 for block in run)
    across = float(np.mean([block[3] for block in run]))
    along = (start + end) / 2.0
    center = (along, across) if horizontal else (across, along)
    return Opening(
        kind="window",
        center=(float(center[0]), float(center[1])),
        width_px=float(end - start),
        wall_id=wall.id,
        horizontal=horizontal,
        # Several merged blocks is the drawn convention for a wide window, so a
        # merged run is stronger evidence than a lone block.
        confidence=0.75 if len(run) > 1 else 0.6,
    )


def _detect_doors(
    thin: np.ndarray, walls: list[Wall], door_span: float, windows: list[Opening],
) -> list[Opening]:
    """Swing arcs: circular, and bulging perpendicular away from their wall."""
    contours, _ = cv2.findContours(thin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    jamb_tolerance = door_span * 1.2
    openings: list[Opening] = []
    for contour in contours:
        if len(contour) < 8:
            continue
        points = contour.reshape(-1, 2).astype(float)
        span = float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
        # A swing arc is about as big as the door it belongs to.
        if not (door_span * 0.45 <= span <= door_span * 2.6):
            continue
        fit = _fit_circle(points)
        if fit is None:
            continue
        cx, cy, radius, residual = fit
        if residual > 0.16 or not (door_span * 0.35 <= radius <= door_span * 2.0):
            continue
        # The hinge sits on a wall, and the arc sweeps away from it.
        wall, horizontal, distance = _nearest_wall(cx, cy, walls, jamb_tolerance)
        if wall is None:
            continue
        if not _bulges_off_wall(points, cx, cy, horizontal, radius):
            continue
        openings.append(Opening(
            kind="door",
            center=(float(cx), float(cy)),
            width_px=float(radius),  # swing radius == leaf length == clear width
            wall_id=wall.id,
            horizontal=horizontal,
            confidence=float(max(0.35, min(0.9, 1.0 - residual * 3.0))),
        ))
    # A window box can weakly imitate an arc; the window classification wins.
    return [
        door for door in openings
        if not any(_close(door.center, window.center, door_span * 0.6) for window in windows)
    ]


def _bulges_off_wall(
    points: np.ndarray, cx: float, cy: float, horizontal: bool, radius: float,
) -> bool:
    """A door arc extends perpendicular to its wall; a window box never does."""
    across = points[:, 1] - cy if horizontal else points[:, 0] - cx
    along = points[:, 0] - cx if horizontal else points[:, 1] - cy
    perpendicular = float(np.abs(across).max())
    parallel = float(np.abs(along).max())
    # Must reach well off the wall line, and not be a sliver lying along it.
    return perpendicular >= radius * 0.45 and perpendicular >= parallel * 0.35


def _fit_circle(points: np.ndarray) -> tuple[float, float, float, float] | None:
    """Algebraic circle fit; returns (cx, cy, radius, normalised residual)."""
    x, y = points[:, 0], points[:, 1]
    try:
        solution, *_ = np.linalg.lstsq(
            np.c_[2 * x, 2 * y, np.ones(len(points))], x ** 2 + y ** 2, rcond=None,
        )
    except np.linalg.LinAlgError:
        return None
    cx, cy = float(solution[0]), float(solution[1])
    radii = np.hypot(x - cx, y - cy)
    mean = float(radii.mean())
    if mean < 1e-6:
        return None
    return cx, cy, mean, float(radii.std() / mean)


def _close(a: Point, b: Point, tolerance: float) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance


def _dedupe(openings: list[Opening], min_separation: float) -> list[Opening]:
    kept: list[Opening] = []
    for opening in sorted(openings, key=lambda item: -item.confidence):
        if any(item.kind == opening.kind and _close(item.center, opening.center, min_separation)
               for item in kept):
            continue
        kept.append(opening)
    return kept[:120]
