from slm.structured import EXTRACT_PROMPT, Extraction, extract


def test_stream_yields_tokens_then_stats(mock_ollama):
    client = mock_ollama(["Hel", "lo"])
    events = list(client.stream("m", "hi", fmt="json", system="sys", seed=7, num_predict=5))
    assert [e.text for e in events[:-1]] == ["Hel", "lo"] and events[-1].done
    assert events[-1].stats["eval_count"] == 2 and "response" not in events[-1].stats
    body = mock_ollama.seen[-1]
    assert body["think"] is False and body["format"] == "json" and body["system"] == "sys"
    assert body["options"] == {"temperature": 0.0, "num_predict": 5, "seed": 7}
    text, stats = client.generate("m", "hi")
    assert text == "Hello" and stats["prompt_eval_count"] == 12


class Scripted:
    """Returns the scripted raw outputs in order."""

    def __init__(self, *outs):
        self.outs, self.prompts, self.fmts = list(outs), [], []

    def generate(self, model, prompt, fmt=None, **kw):
        self.prompts.append(prompt)
        self.fmts.append(fmt)
        return self.outs.pop(0), {}


GOOD = '{"intent": "command", "entities": ["Sara", "thursday"], "confidence": 0.8}'


def test_valid_first_time():
    a = extract(Scripted(GOOD), "m", "move my 3pm with Sara to thursday")
    assert a.attempts == 1 and a.result == Extraction(intent="command", entities=["Sara", "thursday"], confidence=0.8)


def test_one_repair_with_the_validation_error():
    c = Scripted('{"intent": "request", "entities": [], "confidence": 2}', GOOD)
    a = extract(c, "m", "x")
    assert a.attempts == 2 and a.result is not None and len(a.raw) == 2
    assert "Validation error" in c.prompts[1] and "intent" in c.prompts[1]


def test_fails_visibly_after_the_retry():
    a = extract(Scripted("not json", "{}"), "m", "x")
    assert a.result is None and a.attempts == 2 and a.raw == ["not json", "{}"]


def test_schema_mode_passes_the_json_schema():
    c = Scripted(GOOD)
    extract(c, "m", "x", schema=True)
    assert c.fmts[0]["properties"]["intent"]["enum"] == ["question", "command", "chitchat"]
    assert "{text}" not in EXTRACT_PROMPT.format(text="x")
