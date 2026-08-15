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
from .correction import TopologyCorrectionPlanner, apply_corrections


@dataclass
class GuidanceAudit:
    specification_status: str = "not_run"
    specification_confidence: float = 0.0
    specification_summary: str = ""
    initial_review_status: str = "not_run"
    review_status: str = "not_run"
    review_confidence: float = 0.0
    review_summary: str = ""
    discrepancies: list[dict] = field(default_factory=list)
    accepted: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    correction_status: str = "not_run"
    correction_model: str = ""


class QwenTopologyGuide:
    """Use two vision passes around deterministic, pixel-validated reconstruction."""

    def __init__(self, api_key: str | None = None, model: str = "qwen/qwen3.6-27b",
                 enable_correction: bool = True, building_profile: dict | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.enable_correction = enable_correction
        self.building_profile = building_profile or {}
        self.audit = GuidanceAudit()
        self.plan_spec: dict = {}
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")

    def __call__(self, rgb: np.ndarray, model: FloorplanModel) -> FloorplanModel:
        payload = review_against_spec(self.plan_spec, model)
        self._capture_review(payload, initial=True)
        if not self.enable_correction:
            self.audit.correction_status = "disabled"
            return model
        try:
            planner = TopologyCorrectionPlanner(self.api_key)
            discrepancies = list(self.audit.discrepancies)
            if not discrepancies:
                discrepancies.append({
                    "kind": "independent_visual_verification",
                    "description": (
                        "Independently verify the full topology and its closed exterior perimeter; "
                        "return no operations when it already matches the image."
                    ),
                })
            proposals = planner.propose(rgb, model, discrepancies)
            model = apply_corrections(rgb, model, proposals, self.audit.accepted, self.audit.rejected)
            self.audit.correction_model = planner.model_used
            changed = any(
                item.get("kind") in {"remove_wall", "replace_wall", "add_wall"}
                for item in self.audit.accepted
            )
            self.audit.correction_status = "applied" if changed else "verified_no_change"
            revised = review_against_spec(self.plan_spec, model)
            self._capture_review(revised, initial=False)
        except Exception as exc:
            self.audit.correction_status = "failed"
            self.audit.rejected.append({"kind": "correction_stage", "reason": str(exc)[:500]})
        return model

    def specify(self, rgb: np.ndarray) -> dict:
        """Create the vision-first semantic target that constrains reconstruction."""
        height, width = rgb.shape[:2]
        prompt = (
            "Inspect this floor-plan image before any geometry extraction. Define what a deterministic Python "
            "reconstructor should produce. Coordinates are pixels in image_size. Return JSON only: "
            '{"status":"usable|insufficient","confidence":0..1,"summary":string,'
            '"spaces":[{"label":string,"type":string,"category":"occupied|circulation|service|shaft|safety",'
            '"center":[x,y],"bbox":[x0,y0,x1,y1],'
            '"confidence":0..1}],"major_walls":[{"orientation":"horizontal|vertical",'
            '"start":[x,y],"end":[x,y],"kind":"external|internal","confidence":0..1}],'
            '"openings":[{"type":"door|window","center":[x,y],"confidence":0..1}]}. '
            'Also return "grid":{"x":[pixel positions],"y":[pixel positions]} and '
            '"elements":[{"id":string,"kind":string,"category":"structural|circulation|safety|mep|other",'
            '"label":string,"center":[x,y],"bbox":[x0,y0,x1,y1],"confidence":0..1}]. '
            "Include commercial spaces and features when visible: offices, tenant areas, corridors, lobbies, "
            "stairs, fire-escape stairs, lift banks, shafts, service/electrical rooms, fire command centres, "
            "refuge areas, compartments and exits. Preserve exact visible labels. Do not assume residential "
            "space types or invent features from the building profile. Include only structural walls, not "
            "furniture, dimension lines, text underlines, or room contents. "
            'Also read every printed dimension label so real-world measurements can be recovered (EasyOCR is '
            'not available at runtime, so this vision read is the only source for it). Return '
            '"dimensions":[{"value_m":number,"orientation":"horizontal|vertical","position":[x,y]}] — one '
            'entry per printed dimension number along the dimension lines (e.g. each segment like "2.00", '
            '"4.00", "3.00" on a wall run, not just the overall total). "position" is the pixel location of '
            'that label, placed near the wall segment it measures. "orientation" is horizontal for a '
            'width/length label, vertical for a height/depth label. Convert feet/inches labels to metres. '
            'Read as many dimension labels as are legibly printed; omit ones you cannot read confidently. Also '
            'return "overall_width_m":number,"overall_height_m":number set to the printed OUTER envelope '
            'dimension labels in metres (the outermost total, not a single room). Set both to 0 if no explicit '
            'overall total is printed — never estimate or guess a value. '
            f"image_size=[{width},{height}]. Building profile context="
            + json.dumps(self.building_profile)
        )
        self.plan_spec = _qwen_json(self.api_key, self.model, rgb, prompt, max_tokens=1800)
        status = str(self.plan_spec.get("status", "insufficient"))
        self.audit.specification_status = status if status in {"usable", "insufficient"} else "insufficient"
        self.audit.specification_confidence = _bounded_confidence(self.plan_spec.get("confidence", 0))
        self.audit.specification_summary = str(self.plan_spec.get("summary", ""))[:500]
        return self.plan_spec

    def _capture_review(self, payload: dict, initial: bool) -> None:
        review = payload.get("review") if isinstance(payload.get("review"), dict) else {}
        status = str(review.get("status", "insufficient"))
        status = status if status in {"approved", "needs_correction", "insufficient"} else "insufficient"
        if initial:
            self.audit.initial_review_status = status
        self.audit.review_status = status
        self.audit.review_confidence = _bounded_confidence(review.get("confidence", 0))
        self.audit.review_summary = str(review.get("summary", ""))[:500]
        self.audit.discrepancies = list(review.get("discrepancies") or [])[:50]

    def _request(self, rgb: np.ndarray, model: FloorplanModel) -> dict:
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
        return _qwen_json(self.api_key, self.model, rgb, prompt, max_tokens=1800)


def review_against_spec(spec: dict, model: FloorplanModel) -> dict:
    """Measure Python topology against Qwen's pixel-coordinate target without another vision call."""
    if not isinstance(spec, dict) or spec.get("status") != "usable":
        return {"review": {"status": "insufficient",
                           "confidence": _bounded_confidence(spec.get("confidence", 0) if isinstance(spec, dict) else 0),
                           "summary": "Vision specification was insufficient for geometry approval.",
                           "discrepancies": []}}
    width, height = model.image_size
    tolerance = max(8.0, max(width, height) * 0.025)
    walls = [item for item in list(spec.get("major_walls") or [])
             if _bounded_confidence(item.get("confidence", 0)) >= 0.65]
    spaces = [item for item in list(spec.get("spaces") or [])
              if _bounded_confidence(item.get("confidence", 0)) >= 0.65]
    discrepancies = []
    matched = 0
    exterior_matches = []
    for item in walls:
        matching_wall = next(
            (wall for wall in model.walls if _wall_matches(item, wall, tolerance)),
            None,
        )
        if matching_wall is not None:
            matched += 1
            if item.get("kind") == "external":
                exterior_matches.append(matching_wall)
        else:
            discrepancies.append({"kind": "missing_specified_wall",
                                  "description": f"No pixel-supported reconstructed wall matches {item.get('start')} to {item.get('end')}.",
                                  "target": item})
    for item in spaces:
        center = item.get("center")
        if any(_room_contains(room, center, tolerance) for room in model.rooms):
            matched += 1
        else:
            discrepancies.append({"kind": "missing_specified_space",
                                  "description": f"No reconstructed space contains vision target {item.get('label')!r} at {center}.",
                                  "target": item})
    total = len(walls) + len(spaces)
    ratio = matched / total if total else 0.0
    exterior_targets = [item for item in walls if item.get("kind") == "external"]
    perimeter_closed = not exterior_targets or _closed_perimeter(
        exterior_matches, tolerance
    )
    if exterior_targets and not perimeter_closed:
        discrepancies.append({
            "kind": "open_external_perimeter",
            "description": "Exterior wall segments do not form one unbroken closed perimeter.",
        })
    status = (
        "approved"
        if total and ratio >= 0.70 and perimeter_closed
        else ("needs_correction" if total else "insufficient")
    )
    confidence = _bounded_confidence(spec.get("confidence", 0)) * (0.5 + 0.5 * ratio)
    return {"review": {"status": status, "confidence": confidence,
                       "summary": f"Matched {matched}/{total} high-confidence vision targets to pixel-derived topology.",
                       "discrepancies": discrepancies[:50]}}


def _closed_perimeter(walls: list[Wall], tolerance: float) -> bool:
    if len(walls) < 4:
        return False
    nodes: list[tuple[float, float]] = []
    edges = []

    def node_for(point):
        for index, node in enumerate(nodes):
            if np.hypot(point[0] - node[0], point[1] - node[1]) <= tolerance:
                return index
        nodes.append((float(point[0]), float(point[1])))
        return len(nodes) - 1

    for wall in walls:
        a, b = node_for(wall.start), node_for(wall.end)
        if a != b:
            edges.append((a, b))
    if len(edges) < 4:
        return False
    degrees = [0] * len(nodes)
    adjacency = [set() for _ in nodes]
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
        adjacency[a].add(b)
        adjacency[b].add(a)
    if any(degree != 2 for degree in degrees):
        return False
    visited = set()
    stack = [0]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency[current] - visited)
    return len(visited) == len(nodes)


def _wall_matches(item: dict, wall: Wall, tolerance: float) -> bool:
    try:
        start, end = item["start"], item["end"]
        sx, sy, ex, ey = map(float, (start[0], start[1], end[0], end[1]))
        wx1, wy1 = map(float, wall.start)
        wx2, wy2 = map(float, wall.end)
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    horizontal = abs(ex - sx) >= abs(ey - sy)
    wall_horizontal = abs(wx2 - wx1) >= abs(wy2 - wy1)
    if horizontal != wall_horizontal:
        return False
    if horizontal:
        return abs((sy + ey - wy1 - wy2) / 2) <= tolerance and _overlap(sx, ex, wx1, wx2) >= 0.45
    return abs((sx + ex - wx1 - wx2) / 2) <= tolerance and _overlap(sy, ey, wy1, wy2) >= 0.45


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    alo, ahi = sorted((a1, a2))
    blo, bhi = sorted((b1, b2))
    return max(0.0, min(ahi, bhi) - max(alo, blo)) / max(ahi - alo, 1.0)


def _room_contains(room, center, tolerance: float) -> bool:
    if not isinstance(center, (list, tuple)) or len(center) != 2 or not room.boundary:
        return False
    try:
        x, y = map(float, center)
        xs = [float(point[0]) for point in room.boundary]
        ys = [float(point[1]) for point in room.boundary]
    except (TypeError, ValueError, IndexError):
        return False
    return min(xs) - tolerance <= x <= max(xs) + tolerance and min(ys) - tolerance <= y <= max(ys) + tolerance

def _qwen_json(api_key: str, model: str, rgb: np.ndarray, prompt: str, max_tokens: int) -> dict:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("could not encode plan image")
        image_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        result = None
        for attempt in range(2):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    " Previous generation failed JSON validation. Return exactly one complete JSON object "
                    "matching the requested keys, with no markdown, comments, or trailing text."
                )
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": attempt_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}],
                "temperature": 0.7,
                "top_p": 0.8,
                "presence_penalty": 1.5,
                "max_completion_tokens": max_tokens,
                "reasoning_effort": "none",
                "response_format": {"type": "json_object"},
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "FireShield-Floorplan/1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                provider_code, provider_message = _provider_error_details(exc)
                if exc.code == 400 and provider_code == "json_validate_failed" and attempt == 0:
                    continue
                suffix = f": {provider_message}" if provider_message else ""
                raise RuntimeError(f"Qwen request failed ({exc.code}){suffix}") from exc
        if result is None:
            raise RuntimeError("Qwen request failed after JSON validation retry")
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return json.loads(content)


def _bounded_confidence(value) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _provider_error_message(error: urllib.error.HTTPError) -> str:
    """Return only Groq's bounded error message/code, never headers or keys."""
    _, message = _provider_error_details(error)
    return message


def _provider_error_details(error: urllib.error.HTTPError) -> tuple[str, str]:
    """Return Groq's bounded error code and display message."""
    try:
        payload = json.loads(error.read(4096).decode("utf-8", errors="replace"))
        detail = payload.get("error", payload) if isinstance(payload, dict) else {}
        if not isinstance(detail, dict):
            return "", ""
        message = str(detail.get("message", ""))[:500]
        code = str(detail.get("code", ""))[:100]
        display = f"{code}: {message}" if code and code not in message else message
        return code, display
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "", ""


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
