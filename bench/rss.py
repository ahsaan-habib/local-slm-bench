"""Peak memory of the model, not of this script.

Ollama runs inference in a separate runner process, so tracemalloc or our own
RSS tells you nothing. Sample the RSS of every ollama process in a thread and
keep the max; ask /api/ps how much of the model sits in VRAM.
"""
from __future__ import annotations

import threading

import httpx
import psutil

from slm.client import OLLAMA_URL


def _ollama_rss() -> int:
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        name = (p.info.get("name") or "").lower()
        if "ollama" in name and p.info.get("memory_info"):
            total += p.info["memory_info"].rss
    return total


class RSSWatcher:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _ollama_rss())
            self._stop.wait(self.interval)

    def start(self) -> "RSSWatcher":
        self.peak, self._stop = _ollama_rss(), threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._t:
            self._t.join()
        return round(self.peak / 2**20, 1)


def loaded_model(model: str) -> dict:
    """{'size_mb': .., 'vram_mb': .., 'fully_in_vram': bool} for a loaded model."""
    r = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=10)
    r.raise_for_status()
    for m in r.json().get("models", []):
        if m["name"] == model or m["model"] == model:
            size, vram = m.get("size", 0), m.get("size_vram", 0)
            return {"size_mb": round(size / 2**20), "vram_mb": round(vram / 2**20),
                    "fully_in_vram": bool(size) and vram >= size}
    return {"size_mb": 0, "vram_mb": 0, "fully_in_vram": False}
