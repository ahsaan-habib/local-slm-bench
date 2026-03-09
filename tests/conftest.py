"""Offline by default: Ollama is replaced by httpx.MockTransport or small
fakes. test_smoke_ollama.py talks to a real model and is opt-in."""
import json

import httpx
import pytest

from slm.client import Ollama


def ndjson(*objs) -> bytes:
    return "".join(json.dumps(o) + "\n" for o in objs).encode()


@pytest.fixture
def mock_ollama():
    """An Ollama client whose /api/generate streams `tokens`, then a done line."""
    seen = []

    def make(tokens, stats=None):
        def handler(request: httpx.Request):
            seen.append(json.loads(request.content))
            lines = [{"response": t, "done": False} for t in tokens]
            lines.append({"response": "", "done": True, "prompt_eval_count": 12, "eval_count": len(tokens),
                          "eval_duration": 1_000_000_000, "prompt_eval_duration": 5_000_000, **(stats or {})})
            return httpx.Response(200, content=ndjson(*lines))

        client = Ollama()
        client.http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
        return client

    make.seen = seen
    return make
