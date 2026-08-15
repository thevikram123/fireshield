"""OpenCV topology: complete wall faces, interior rooms, envelope, dimension bars."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .schema import Point, Room, VectorTrace, Wall


@dataclass
class DimensionBar:
    start: Point
    end: Point
    length_px: float
    horizontal: bool


@dataclass
class CvGeometry:
    walls: list[Wall] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    envelope: tuple[float, float, float, float] | None = None  # x0,y0,x1,y1
    bars: list[DimensionBar] = field(default_factory=list)
    traces: list[VectorTrace] = field(default_factory=list)
    wall_mask: np.ndarray | None = None


def extract_geometry(rgb: np.ndarray, wall_mask: np.ndarray) -> CvGeometry:
    raw = wall_mask if wall_mask is not None else np.zeros(rgb.shape[:2], np.uint8)
    h, w = raw.shape[:2]
    dash = max(9, int(round(min(h, w) * 0.022)))
    door = max(17, int(round(min(h, w) * 0.045)))
    linked = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _kernel(dash), iterations=1)
    ink = keep_building_ink(linked)
    main = largest_structure(ink)
    structure = clip_to_outer(cv2.bitwise_or(ink, seal_outer_loop(main, thickness=max(8, dash))), main)
    sealed = cv2.morphologyEx(structure, cv2.MORPH_CLOSE, _kernel(door), iterations=1)
    walls = walls_from_mask(cv2.bitwise_or(ink, structure))
    rooms = rooms_from_mask(sealed)
    envelope = envelope_from_mask(structure)
    bars = find_dimension_bars(rgb)
    traces = trace_all_lines(rgb)
    return CvGeometry(
        walls=walls, rooms=rooms, envelope=envelope, bars=bars,
        traces=traces, wall_mask=structure,
    )


def _kernel(n: int) -> np.ndarray:
    n = max(3, int(n) | 1)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def clip_to_outer(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    fill = np.zeros_like(mask)
    dilated = cv2.dilate(seed, _kernel(15), iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    cv2.drawContours(fill, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=-1)
    return cv2.bitwise_and(mask, fill)


def largest_structure(mask: np.ndarray) -> np.ndarray:
    work = (mask > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    h = work.shape[0]
    best_i, best_score = 0, 0.0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cy = int(stats[i, cv2.CC_STAT_TOP]) + hh / 2.0
        if hh < 25 and ww > 80 and cy > 0.78 * h:
            continue
        if area < 80:
            continue
        score = float(ww * hh)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i == 0:
        return work
    return np.where(labels == best_i, 255, 0).astype(np.uint8)


def keep_building_ink(mask: np.ndarray) -> np.ndarray:
    """Drop title underlines and speckle; keep every other wall fragment."""
    work = (mask > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    h = work.shape[0]
    keep = np.zeros_like(work)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cy = int(stats[i, cv2.CC_STAT_TOP]) + hh / 2.0
        if area < 40:
            continue
        if hh < 25 and ww > 80 and cy > 0.78 * h:
            continue
        keep[labels == i] = 255
    return keep if cv2.countNonZero(keep) else work


def seal_outer_loop(mask: np.ndarray, thickness: int = 10) -> np.ndarray:
    """Add a closed outer belt so window dashes cannot leak the interior."""
    work = (mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(work) < 20:
        return work
    pad = max(9, thickness)
    dilated = cv2.dilate(work, _kernel(pad), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return work
    outer = max(contours, key=cv2.contourArea)
    fill = np.zeros_like(work)
    cv2.drawContours(fill, [outer], -1, 255, thickness=-1)
    belt = max(7, thickness)
    inner = cv2.erode(fill, _kernel(belt), iterations=1)
    ring = cv2.subtract(fill, inner)
    return cv2.bitwise_or(work, ring)


def walls_from_mask(mask: np.ndarray, min_area: int = 40) -> list[Wall]:
    if mask is None or mask.size == 0:
        return []
    work = (mask > 0).astype(np.uint8) * 255
    h, w = work.shape[:2]
    walls: list[Wall] = []
    min_len = max(22, int(round(min(h, w) * 0.04)))
    dist = cv2.distanceTransform(work, cv2.DIST_L2, 3)
    # Keep both filled bands and thin line evidence. Classification happens
    # after extraction so doors, partitions and dimension evidence are not lost.
    band_segments = _trace_wall_bands(work, min_len)
    hough = cv2.HoughLinesP(
        work, 1, np.pi / 180.0, threshold=28,
        minLineLength=max(14, min_len // 2), maxLineGap=max(8, min_len // 3),
    )
    line_segments = (
        [tuple(int(value) for value in segment) for segment in np.asarray(hough).reshape(-1, 4)]
        if hough is not None else []
    )
    merged = _complete_junctions(
        _merge_segments(band_segments + line_segments, min_len=max(14, min_len // 2)),
        tolerance=max(6, min_len // 4),
    )
    for i, (x1, y1, x2, y2) in enumerate(merged, start=len(walls)):
        thick = _sample_thickness(dist, x1, y1, x2, y2)
        walls.append(
            Wall(
                id=f"wall_{i}",
                start=(float(x1), float(y1)),
                end=(float(x2), float(y2)),
                thickness_mm=float(max(thick * 2.0, 2.0)),
                wall_type="thick_candidate" if thick * 2.0 >= 5.0 else "thin_candidate",
                quad=[(float(x1), float(y1)), (float(x2), float(y2))],
            )
        )
    if not walls:
        walls.extend(_contour_walls(work, start_id=len(walls)))
    return walls


def _trace_wall_bands(mask: np.ndarray, min_len: int) -> list[tuple[int, int, int, int]]:
    segments: list[tuple[int, int, int, int]] = []
    for horizontal in (True, False):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (min_len, 3) if horizontal else (3, min_len)
        )
        bands = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(bands, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            span = width if horizontal else height
            thickness = height if horizontal else width
            if span < min_len or thickness > max(30, min_len):
                continue
            if horizontal:
                cy = y + height // 2
                segments.append((x, cy, x + width - 1, cy))
            else:
                cx = x + width // 2
                segments.append((cx, y, cx, y + height - 1))
    return _merge_segments(segments, min_len=min_len)


def _complete_junctions(
    segments: list[tuple[int, int, int, int]], tolerance: int
) -> list[tuple[int, int, int, int]]:
    """Close small wall gaps by snapping endpoints to traced perpendicular bands."""
    mutable = [list(segment) for segment in segments]
    horizontals = [item for item in mutable if item[1] == item[3]]
    verticals = [item for item in mutable if item[0] == item[2]]
    for horizontal in horizontals:
        x0, y, x1, _ = horizontal
        for vertical in verticals:
            x, y0, _, y1 = vertical
            if y0 - tolerance <= y <= y1 + tolerance and x0 - tolerance <= x <= x1 + tolerance:
                if abs(x0 - x) <= tolerance:
                    horizontal[0] = x
                if abs(x1 - x) <= tolerance:
                    horizontal[2] = x
                if abs(y0 - y) <= tolerance:
                    vertical[1] = y
                if abs(y1 - y) <= tolerance:
                    vertical[3] = y
    return [tuple(item) for item in mutable]


def _silhouette(mask: np.ndarray) -> np.ndarray:
    if cv2.countNonZero(mask) < 20:
        return mask
    dilated = cv2.dilate(mask, _kernel(15), iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(fill, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=-1)
    return fill


def _outer_ring(silhouette: np.ndarray) -> list[tuple[float, float]]:
    cnts, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, max(2.0, 0.004 * cv2.arcLength(cnt, True)), True)
    if len(approx) < 3:
        return []
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def _merge_segments(segs: list[tuple[int, int, int, int]], min_len: int) -> list[tuple[int, int, int, int]]:
    horiz: dict[int, list[tuple[int, int]]] = {}
    vert: dict[int, list[tuple[int, int]]] = {}
    for x1, y1, x2, y2 in segs:
        if abs(x2 - x1) >= abs(y2 - y1):
            y = int(round((y1 + y2) / 2.0))
            a, b = sorted((x1, x2))
            horiz.setdefault(y, []).append((a, b))
        else:
            x = int(round((x1 + x2) / 2.0))
            a, b = sorted((y1, y2))
            vert.setdefault(x, []).append((a, b))
    out: list[tuple[int, int, int, int]] = []
    gap = max(12, min_len // 2)
    bin_px = 12
    out.extend(_merge_axis(horiz, True, gap, bin_px, min_len))
    out.extend(_merge_axis(vert, False, gap, bin_px, min_len))
    return out


def _merge_axis(buckets, horizontal: bool, gap: int, bin_px: int, min_len: int):
    if not buckets:
        return []
    items = []
    for key, spans in buckets.items():
        items.append((key, list(spans)))
    items.sort()
    clusters: list[tuple[float, list[tuple[int, int]]]] = []
    for key, spans in items:
        if clusters and abs(key - clusters[-1][0]) <= bin_px:
            prev_key, prev_spans = clusters[-1]
            clusters[-1] = ((prev_key + key) / 2.0, prev_spans + spans)
        else:
            clusters.append((float(key), spans))
    merged = []
    for key, spans in clusters:
        spans = sorted(spans)
        cur_a, cur_b = spans[0]
        acc = [(cur_a, cur_b)]
        for a, b in spans[1:]:
            if a <= cur_b + gap:
                cur_b = max(cur_b, b)
                acc[-1] = (acc[-1][0], cur_b)
            else:
                cur_a, cur_b = a, b
                acc.append((cur_a, cur_b))
        pos = int(round(key))
        for a, b in acc:
            if b - a < min_len:
                continue
            if horizontal:
                merged.append((a, pos, b, pos))
            else:
                merged.append((pos, a, pos, b))
    return merged


def _sample_thickness(dist: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    h, w = dist.shape[:2]
    vals = []
    for t in (0.25, 0.5, 0.75):
        x = int(round(x1 + t * (x2 - x1)))
        y = int(round(y1 + t * (y2 - y1)))
        if 0 <= x < w and 0 <= y < h:
            vals.append(float(dist[y, x]))
    return max(vals) if vals else 2.0


def _contour_walls(mask: np.ndarray, start_id: int = 0) -> list[Wall]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    extra: list[Wall] = []
    for i, cnt in enumerate(cnts):
        if abs(cv2.contourArea(cnt)) < 80:
            continue
        approx = cv2.approxPolyDP(cnt, max(1.5, 0.006 * cv2.arcLength(cnt, True)), True)
        if len(approx) < 3:
            continue
        ring = [(float(p[0][0]), float(p[0][1])) for p in approx]
        extra.append(
            Wall(
                id=f"wall_{start_id + i}",
                start=ring[0],
                end=ring[len(ring) // 2],
                thickness_mm=6.0,
                quad=ring,
            )
        )
    return extra


def rooms_from_mask(mask: np.ndarray, min_frac: float = 0.004) -> list[Room]:
    if mask is None or mask.size == 0:
        return []
    work = (mask > 0).astype(np.uint8) * 255
    h, w = work.shape[:2]
    interior = cv2.bitwise_not(work)
    flood = interior.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[seed[1], seed[0]] != 0:
            cv2.floodFill(flood, ff_mask, seed, 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(flood, connectivity=8)
    min_area = max(80.0, min_frac * h * w)
    max_area = 0.85 * h * w
    rooms: list[Room] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(1.5, 0.006 * peri), True)
        if len(approx) < 3:
            continue
        ring = [(float(p[0][0]), float(p[0][1])) for p in approx]
        rooms.append(Room(id=f"room_{len(rooms)}", type="UNDEFINED", boundary=ring))
    return rooms


def envelope_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    if mask is None or cv2.countNonZero(mask) < 20:
        return None
    ys, xs = np.where(mask > 0)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def find_dimension_bars(rgb: np.ndarray) -> list[DimensionBar]:
    """Long colored dimension ticks (any hue). Black wall ink is ignored."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    color = cv2.inRange(hsv, (0, 45, 70), (180, 255, 255))
    color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    lines = cv2.HoughLinesP(color, 1, np.pi / 180.0, threshold=40, minLineLength=60, maxLineGap=12)
    if lines is None or len(lines) == 0:
        return []
    horiz: list[DimensionBar] = []
    vert: list[DimensionBar] = []
    for seg in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (int(v) for v in seg)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 60:
            continue
        horizontal = abs(x2 - x1) >= abs(y2 - y1)
        bar = DimensionBar(
            start=(float(x1), float(y1)),
            end=(float(x2), float(y2)),
            length_px=length,
            horizontal=horizontal,
        )
        (horiz if horizontal else vert).append(bar)
    bars: list[DimensionBar] = []
    if horiz:
        bars.append(max(horiz, key=lambda b: b.length_px))
    if vert:
        bars.append(max(vert, key=lambda b: b.length_px))
    return bars


def trace_all_lines(rgb: np.ndarray) -> list[VectorTrace]:
    """Vectorize every meaningful visible ink contour without semantic filtering."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    otsu = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 11,
    )
    ink = cv2.bitwise_or(otsu, adaptive)
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    traces: list[VectorTrace] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 8:
            continue
        approx = cv2.approxPolyDP(contour, max(0.6, perimeter * 0.0015), True)
        points = [(float(item[0][0]), float(item[0][1])) for item in approx]
        if len(points) < 2:
            continue
        _, (box_width, box_height), _ = cv2.minAreaRect(contour)
        stroke_band = min(box_width, box_height)
        traces.append(VectorTrace(
            id=f"trace_{len(traces)}", points=points,
            closed=bool(len(points) >= 3), line_class="thick" if stroke_band >= 4 else "thin",
        ))
    return traces[:3000]
