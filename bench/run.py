"""bench/run.py — CSV, because screenshots are not data."""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from slm.client import Ollama

from .rss import RSSWatcher, loaded_model


@dataclass
class Sample:
    model: str
    prompt_id: str
    run: int
    ttft_ms: float
    decode_tps: float       # tokens after the first / time after the first
    total_ms: float
    output_tokens: int
    input_tokens: int
    server_prefill_ms: float  # ollama's own prompt_eval_duration
    server_decode_tps: float  # eval_count / eval_duration
    peak_rss_mb: float
    vram_mb: int
    fully_in_vram: bool


def measure(client: Ollama, model: str, prompt_id: str, prompt: str, run: int) -> Sample:
    watcher = RSSWatcher().start()
    t0 = time.perf_counter()
    first = None
    tokens = 0
    stats: dict = {}
    for ev in client.stream(model, prompt, temperature=0.0):
        if ev.done:
            stats = ev.stats
            break
        if first is None:
            first = time.perf_counter()  # TTFT: the first token, not the first byte
        tokens += 1
    end = time.perf_counter()
    peak = watcher.stop()
    mem = loaded_model(model)
    return Sample(
        model=model, prompt_id=prompt_id, run=run,
        ttft_ms=round((first - t0) * 1000, 1),
        # NOT tokens / total: that folds prefill into the decode rate and makes
        # every model look slower on long prompts, by a different amount each
        decode_tps=round((tokens - 1) / (end - first), 2) if tokens > 1 else 0.0,
        total_ms=round((end - t0) * 1000, 1),
        output_tokens=tokens,
        input_tokens=stats.get("prompt_eval_count", 0),
        server_prefill_ms=round(stats.get("prompt_eval_duration", 0) / 1e6, 1),
        server_decode_tps=round(stats.get("eval_count", 0) / (stats.get("eval_duration", 0) / 1e9), 2)
        if stats.get("eval_duration") else 0.0,
        peak_rss_mb=peak,
        vram_mb=mem["vram_mb"],
        fully_in_vram=mem["fully_in_vram"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3:4b"])
    ap.add_argument("--suite", default="prompts/suite.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="results/bench.csv")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in Path(args.suite).read_text().splitlines() if l.strip()]
    client = Ollama()
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(Sample.__dataclass_fields__))
        w.writeheader()
        for model in args.models:
            for run in range(args.runs):
                for p in prompts:
                    s = measure(client, model, p["id"], p["prompt"], run)
                    w.writerow(asdict(s))
                    print(f"{model:<16} {p['id']:<8} run {run}  ttft {s.ttft_ms:>7.1f} ms  {s.decode_tps:>6.1f} t/s")


if __name__ == "__main__":
    main()
