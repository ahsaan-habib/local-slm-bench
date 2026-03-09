from fastapi.testclient import TestClient

from slm import api
from slm.client import StreamEvent
from slm.structured import Attempted, Extraction


def test_generate_streams_plain_text(monkeypatch):
    def stream(model, prompt, **kw):
        yield StreamEvent("Hi", False)
        yield StreamEvent(" there", False)
        yield StreamEvent("", True, {"eval_count": 2})
    monkeypatch.setattr(api.client, "stream", stream)
    r = TestClient(api.app).post("/generate", json={"prompt": "hello"})
    assert r.status_code == 200 and r.text == "Hi there"


def test_extract_ok_and_422(monkeypatch):
    c = TestClient(api.app)
    good = Extraction(intent="question", entities=[], confidence=0.9)
    monkeypatch.setattr(api, "extract", lambda *a, **k: Attempted(good, 1, ["{}"]))
    assert c.post("/extract", json={"text": "x"}).json()["intent"] == "question"
    monkeypatch.setattr(api, "extract", lambda *a, **k: Attempted(None, 2, ["a", "b"]))
    r = c.post("/extract", json={"text": "x"})
    assert r.status_code == 422 and r.json()["detail"]["raw"] == ["a", "b"]
    assert c.get("/health").json()["ok"] is True
