"""Against a real local model. Opt-in:

    RUN_OLLAMA=1 pytest tests/test_smoke_ollama.py -s
"""
import os

import pytest

from slm.client import Ollama
from slm.structured import extract

pytestmark = pytest.mark.skipif(os.environ.get("RUN_OLLAMA") != "1", reason="set RUN_OLLAMA=1 to run")
MODEL = os.environ.get("SLM_MODEL", "qwen3:4b-instruct")


def test_stream_and_extract():
    client = Ollama()
    events = list(client.stream(MODEL, "Reply with the single word: pong", num_predict=10))
    text = "".join(e.text for e in events if not e.done)
    assert "pong" in text.lower() and events[-1].stats.get("eval_count", 0) > 0
    a = extract(client, MODEL, "can you move my 3pm with Sara to thursday")
    print("\n", a)
    assert a.result is not None and a.result.intent == "command"
