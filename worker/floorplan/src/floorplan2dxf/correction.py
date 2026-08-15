"""Model-proposed, pixel-validated corrections for floor-plan topology."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np

from .schema import FloorplanModel, Wall


OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
GROQ_FALLBACK_MODEL = "openai/gpt-oss-120b"


def constrain_geometry_to_spec(cv_geom, plan_spec: dict, image_size: tuple[int, int]) -> dict:
    """Match deterministic candidates to a high-confidence vision specification."""
    if (str(plan_spec.get("status")) != "usable"
            or _confidence(plan_spec) < 0.7):
        return {"applied": False, "reason": "vision specification is insufficient"}

    width, height = image_size
    scale = max(width, height, 1)
    predicted = [item for item in list(plan_spec.get("major_walls") or [])[:32]
                 if _confidence(item) >= 0.65]
    candidates = list(cv_geom.walls)
    outline = [wall for wall in candidates if len(wall.quad) >= 3][:1]
    line_candidates = [wall for wall in candidates if wall not in outline]
    matched: list[Wall] = []

    for target in predicted:
        start, end = _axis_segment(target.get("start"), target.get("end"), image_size)
        if not start:
            continue
        target_horizontal = abs(end[0] - start[0]) >= abs(end[1] - start[1])
        best, best_score = None, float("inf")
        tx, ty = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        for wall in line_candidates:
            wall_horizontal = abs(wall.end[0] - wall.start[0]) >= abs(wall.end[1] - wall.start[1])
            if wall_horizontal != target_horizontal:
                continue
            wx, wy = (wall.start[0] + wall.end[0]) / 2, (wall.start[1] + wall.end[1]) / 2
            perpendicular = abs(wy - ty) if target_horizontal else abs(wx - tx)
            along = abs(wx - tx) if target_horizontal else abs(wy - ty)
            score = perpendicular / scale + 0.25 * along / scale
            if score < best_score:
                best, best_score = wall, score
        if best is not None and best_score <= 0.12 and best not in matched:
            best.wall_type = str(target.get("kind", "unknown")) if target.get("kind") in {
                "external", "internal"
            } else "unknown"
            matched.append(best)

    # A usable vision spec is allowed to suppress unmatched Hough lines, which
    # are commonly furniture, text underlines or dimension strokes.
    cv_geom.walls = outline + matched

    expected_rooms = [item for item in list(plan_spec.get("spaces") or plan_spec.get("rooms") or [])[:80]
                      if _confidence(item) >= 0.6]
    selected_rooms = []
    used = set()
    for target in expected_rooms:
        try:
            center = (float(target["center"][0]), float(target["center"][1]))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        best_i, best_distance = None, float("inf")
        for i, room in enumerate(cv_geom.rooms):
            if i in used or not room.boundary:
                continue
            xs, ys = [p[0] for p in room.boundary], [p[1] for p in room.boundary]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            contains = min(xs) <= center[0] <= max(xs) and min(ys) <= center[1] <= max(ys)
            distance = 0.0 if contains else np.hypot(cx - center[0], cy - center[1]) / scale
            if distance < best_distance:
                best_i, best_distance = i, distance
        if best_i is not None and best_distance <= 0.18:
            room = cv_geom.rooms[best_i]
            room.label = str(target.get("label", ""))[:80] or None
            room.type = str(target.get("type", "UNDEFINED"))[:40].upper()
            selected_rooms.append(room)
            used.add(best_i)
    if selected_rooms:
        cv_geom.rooms = selected_rooms

    return {
        "applied": True,
        "specifiedWalls": len(predicted),
        "matchedWalls": len(matched),
        "specifiedRooms": len(expected_rooms),
        "matchedRooms": len(selected_rooms),
    }


class TopologyCorrectionPlanner:
    def __init__(self, openrouter_key: str, groq_key: str):
        self.openrouter_key = openrouter_key
        self.groq_key = groq_key
        self.model_used = ""

    def propose(self, rgb: np.ndarray, model: FloorplanModel, discrepancies: list[dict]) -> dict:
        prompt = _correction_prompt(model, discrepancies)
        if self.openrouter_key:
            try:
                result = self._openrouter(rgb, prompt)
                self.model_used = str(result.get("model") or OPENROUTER_MODEL)
                return _message_json(result)
            except (RuntimeError, urllib.error.URLError, json.JSONDecodeError):
                pass
        if not self.groq_key:
            raise RuntimeError("no correction model is configured")
        result = self._groq(prompt)
        self.model_used = GROQ_FALLBACK_MODEL
        return _message_json(result)

    def _openrouter(self, rgb: np.ndarray, prompt: str) -> dict:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("could not encode correction image")
        image_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        return _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            self.openrouter_key,
            {
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}],
                "temperature": 0.1,
                "max_tokens": 1600,
                "response_format": {"type": "json_object"},
                "stream": False,
            },
            extra_headers={
                "HTTP-Referer": "https://thevikram123.github.io/fireshield/",
                "X-OpenRouter-Title": "FireShield AI",
            },
        )

    def _groq(self, prompt: str) -> dict:
        text_prompt = (
            "You cannot see the source image. Use only Qwen's visual discrepancies and the supplied topology "
            "to propose conservative candidate operations; Python will reject anything unsupported by pixels. "
            + prompt
        )
        return _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            self.groq_key,
            {
                "model": GROQ_FALLBACK_MODEL,
                "messages": [{"role": "user", "content": text_prompt}],
                "temperature": 0.1,
                "max_completion_tokens": 1200,
                "reasoning_effort": "low",
                "include_reasoning": False,
                "response_format": {"type": "json_object"},
                "stream": False,
            },
        )


def apply_corrections(
    rgb: np.ndarray,
    model: FloorplanModel,
    proposals: dict,
    accepted: list[dict],
    rejected: list[dict],
    min_confidence: float = 0.8,
) -> FloorplanModel:
    walls = {wall.id: wall for wall in model.walls}
    remove_ids: set[str] = set()

    for raw in list(proposals.get("remove_walls") or [])[:64]:
        wall = walls.get(str(raw.get("wall_id", "")))
        confidence = _confidence(raw)
        support = _wall_support(rgb, wall.start, wall.end, wall.thickness_mm) if wall else 1.0
        if wall and wall.wall_type != "external" and confidence >= min_confidence and support < 0.42:
            remove_ids.add(wall.id)
            accepted.append({"kind": "remove_wall", "id": wall.id, "confidence": confidence,
                             "pixelSupport": round(support, 3)})
        else:
            rejected.append({"kind": "remove_wall", "proposal": raw,
                             "reason": "wall missing/external, low confidence, or source pixels support it",
                             "pixelSupport": round(support, 3)})

    replacements: list[Wall] = []
    for raw in list(proposals.get("replace_walls") or [])[:32]:
        wall = walls.get(str(raw.get("wall_id", "")))
        start, end = _axis_segment(raw.get("start"), raw.get("end"), model.image_size)
        confidence = _confidence(raw)
        old_support = _wall_support(rgb, wall.start, wall.end, wall.thickness_mm) if wall else 1.0
        new_support = _wall_support(rgb, start, end, raw.get("thickness", 8)) if start else 0.0
        if (wall and wall.wall_type != "external" and start and confidence >= min_confidence
                and old_support < 0.42 and new_support >= 0.50):
            remove_ids.add(wall.id)
            replacements.append(Wall(wall.id, start, end, max(2.0, min(float(raw.get("thickness", 8)), 80.0))))
            accepted.append({"kind": "replace_wall", "id": wall.id, "confidence": confidence,
                             "oldPixelSupport": round(old_support, 3),
                             "newPixelSupport": round(new_support, 3)})
        else:
            rejected.append({"kind": "replace_wall", "proposal": raw,
                             "reason": "replacement is not supported by source pixels",
                             "oldPixelSupport": round(old_support, 3),
                             "newPixelSupport": round(new_support, 3)})

    additions: list[Wall] = []
    for raw in list(proposals.get("add_walls") or [])[:32]:
        start, end = _axis_segment(raw.get("start"), raw.get("end"), model.image_size)
        confidence = _confidence(raw)
        support = _wall_support(rgb, start, end, raw.get("thickness", 8)) if start else 0.0
        duplicate = start and any(_same_segment(start, end, wall.start, wall.end, 6) for wall in model.walls)
        if start and confidence >= min_confidence and support >= 0.50 and not duplicate:
            wall = Wall(f"wall_{len(model.walls) + len(additions)}", start, end,
                        max(2.0, min(float(raw.get("thickness", 8)), 80.0)))
            additions.append(wall)
            accepted.append({"kind": "add_wall", "id": wall.id, "confidence": confidence,
                             "pixelSupport": round(support, 3)})
        else:
            rejected.append({"kind": "add_wall", "proposal": raw,
                             "reason": "addition is invalid, duplicate, or unsupported by source pixels",
                             "pixelSupport": round(support, 3)})

    model.walls = [wall for wall in model.walls if wall.id not in remove_ids] + replacements + additions
    return model


def _correction_prompt(model: FloorplanModel, discrepancies: list[dict]) -> str:
    topology = {
        "image_size": list(model.image_size),
        "walls": [{"id": wall.id, "start": wall.start, "end": wall.end,
                   "thickness": wall.thickness_mm, "wall_type": wall.wall_type}
                  for wall in model.walls[:160]],
        "spaces": [{"id": room.id, "type": room.type, "label": room.label,
                   "boundary": room.boundary[:24]} for room in model.rooms[:80]],
        "doors": [{"id": door.id, "center": door.center, "width": door.width_mm}
                  for door in model.doors[:100]],
    }
    return (
        "Correct the extracted floor-plan topology using the source image and Qwen review. Coordinates are "
        "pixels. Propose only high-confidence axis-aligned wall operations. Do not emit DXF. Return JSON only: "
        '{"summary":string,"remove_walls":[{"wall_id":string,"confidence":0..1,"reason":string}],'
        '"replace_walls":[{"wall_id":string,"start":[x,y],"end":[x,y],"thickness":number,'
        '"confidence":0..1,"reason":string}],"add_walls":[{"start":[x,y],"end":[x,y],'
        '"thickness":number,"confidence":0..1,"reason":string}]}. '
        "Omit uncertain operations. Qwen discrepancies: " + json.dumps(discrepancies[:30])
        + " Extracted topology: " + json.dumps(topology)
    )


def _post_json(url: str, api_key: str, payload: dict, extra_headers: dict | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "Accept": "application/json", "User-Agent": "FireShield-Floorplan/1.0",
        **(extra_headers or {}),
    }
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read(4096)).get("error", {})
            message = str(detail.get("message", ""))[:500]
        except (json.JSONDecodeError, OSError, AttributeError):
            message = ""
        raise RuntimeError(f"correction model failed ({exc.code}){': ' + message if message else ''}") from exc


def _message_json(result: dict) -> dict:
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    return json.loads(content)


def _confidence(raw: dict) -> float:
    try:
        return max(0.0, min(float(raw.get("confidence", 0)), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _axis_segment(start_raw, end_raw, image_size):
    try:
        start = (float(start_raw[0]), float(start_raw[1]))
        end = (float(end_raw[0]), float(end_raw[1]))
    except (TypeError, ValueError, IndexError):
        return None, None
    width, height = image_size
    if not all((0 <= start[0] <= width, 0 <= end[0] <= width,
                0 <= start[1] <= height, 0 <= end[1] <= height)):
        return None, None
    if abs(start[0] - end[0]) <= 3:
        x = (start[0] + end[0]) / 2
        start, end = (x, start[1]), (x, end[1])
    elif abs(start[1] - end[1]) <= 3:
        y = (start[1] + end[1]) / 2
        start, end = (start[0], y), (end[0], y)
    else:
        return None, None
    return (start, end) if np.hypot(end[0] - start[0], end[1] - start[1]) >= 8 else (None, None)


def _wall_support(rgb: np.ndarray, start, end, thickness) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    line_width = max(3, min(int(round(float(thickness))) + 2, 30))
    cv2.line(mask, tuple(map(round, start)), tuple(map(round, end)), 255, line_width)
    pixels = gray[mask > 0]
    return float(np.mean(pixels < 170)) if pixels.size else 0.0


def _same_segment(a, b, c, d, tolerance: float) -> bool:
    direct = np.hypot(a[0] - c[0], a[1] - c[1]) + np.hypot(b[0] - d[0], b[1] - d[1])
    reverse = np.hypot(a[0] - d[0], a[1] - d[1]) + np.hypot(b[0] - c[0], b[1] - c[1])
    return min(direct, reverse) <= tolerance * 2

