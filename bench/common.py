from __future__ import annotations

import json
import platform
import time
from datetime import datetime
from pathlib import Path

import httpx
import psutil
import yaml

from slm.client import OLLAMA_URL


def load_models(path: str = "models.yaml", only: list[str] | None = None) -> list[dict]:
    models = yaml.safe_load(Path(path).read_text())["models"]
    return [m for m in models if not only or m["tag"] in only]


def load_suite(path: str = "prompts/suite.jsonl", kind: str | None = None) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


def unload(model: str) -> None:
    httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)


def cold_load_ms(model: str) -> float:
    """Unload, then time an empty generate — that's pure load time."""
    unload(model)
    time.sleep(1)
    t0 = time.perf_counter()
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "prompt": "", "stream": False},
                   timeout=300)
    r.raise_for_status()
    return round((time.perf_counter() - t0) * 1000, 1)


def machine() -> dict:
    gpu = ""
    try:
        import subprocess
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30),
        "gpu": gpu,
        "ollama": httpx.get(f"{OLLAMA_URL}/api/version", timeout=5).json().get("version", "?"),
    }


def out_path(kind: str) -> Path:
    Path("results").mkdir(exist_ok=True)
    return Path("results") / f"{kind}-{platform.node()}-{datetime.now():%Y%m%d-%H%M}.csv"
