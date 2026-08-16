"""read_dimensions_via_mistral: a separate provider/quota pool for scale
recovery, so it can never collide with whatever the site/photo audit or plan
compliance reasoning is doing against Groq concurrently (which is what
actually broke the Qwen-only path: two Qwen calls per plan hit the same
budget and the second 429'd).
"""

import io
import json
import urllib.error
import urllib.request

import numpy as np

from floorplan2dxf.guidance import read_dimensions_via_mistral


def _fake_response(payload: dict):
    class _Resp:
        def __enter__(self):
            return io.BytesIO(json.dumps(payload).encode("utf-8"))

        def __exit__(self, *args):
            return False

    return _Resp()


def test_request_shape_matches_mistral_docs_exactly(monkeypatch):
    # Confirmed against Mistral's own "Passing a Base64 Encoded Image"
    # example: image_url is a FLAT STRING ("data:image/jpeg;base64,...."),
    # not OpenAI/Groq's nested {"image_url": {"url": ...}}.
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data)
        return _fake_response({"choices": [{"message": {"content": "OVERALL 9.00 11.00\n"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = read_dimensions_via_mistral(
        np.zeros((10, 10, 3), dtype=np.uint8), "test-key", "read the dimensions",
    )

    assert captured["url"] == "https://api.mistral.ai/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    body = captured["body"]
    assert body["model"] == "mistral-medium-latest"
    content = body["messages"][0]["content"]
    image_part = next(part for part in content if part["type"] == "image_url")
    assert isinstance(image_part["image_url"], str)  # flat string, not {"url": ...}
    assert image_part["image_url"].startswith("data:image/jpeg;base64,")
    assert result["overall_width_m"] == 9.0
    assert result["overall_height_m"] == 11.0


def test_retries_once_on_429_honouring_retry_after(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                "https://api.mistral.ai/v1/chat/completions", 429, "rate limited",
                {"Retry-After": "0.01"}, io.BytesIO(b'{"message":"rate limited"}'),
            )
        return _fake_response({"choices": [{"message": {"content": "2.00 H 100 40\nOVERALL 0 0\n"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = read_dimensions_via_mistral(
        np.zeros((10, 10, 3), dtype=np.uint8), "test-key", "read the dimensions",
    )
    assert calls["n"] == 2
    assert len(result["dimensions"]) == 1


def test_a_second_429_after_the_retry_raises_rather_than_looping_forever(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.mistral.ai/v1/chat/completions", 429, "rate limited",
            {"Retry-After": "0.01"}, io.BytesIO(b'{"message":"rate limited"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    try:
        read_dimensions_via_mistral(np.zeros((10, 10, 3), dtype=np.uint8), "test-key", "prompt")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_a_non_429_error_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            "https://api.mistral.ai/v1/chat/completions", 401, "unauthorized",
            {}, io.BytesIO(b'{"message":"invalid api key"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    try:
        read_dimensions_via_mistral(np.zeros((10, 10, 3), dtype=np.uint8), "bad-key", "prompt")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert calls["n"] == 1
