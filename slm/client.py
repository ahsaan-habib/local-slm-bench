"""Ollama over plain HTTP. Streaming matters here: TTFT only exists if you
see the tokens arrive."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


@dataclass
class StreamEvent:
    text: str
    done: bool
    stats: dict = field(default_factory=dict)  # only on the final event


class Ollama:
    def __init__(self, base_url: str = OLLAMA_URL, timeout: float = 300.0):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def stream(self, model: str, prompt: str, temperature: float = 0.0,
               fmt: str | dict | None = None, system: str | None = None,
               num_predict: int = 512, seed: int | None = None) -> Iterator[StreamEvent]:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if seed is not None:
            body["options"]["seed"] = seed
        if fmt is not None:
            body["format"] = fmt
        if system:
            body["system"] = system
        with self.http.stream("POST", "/api/generate", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    yield StreamEvent("", True, {k: v for k, v in data.items() if k != "response"})
                else:
                    yield StreamEvent(data.get("response", ""), False)

    def generate(self, model: str, prompt: str, **kw) -> tuple[str, dict]:
        parts, stats = [], {}
        for ev in self.stream(model, prompt, **kw):
            if ev.done:
                stats = ev.stats
            else:
                parts.append(ev.text)
        return "".join(parts), stats
