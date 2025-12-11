"""bench/run.py — CSV, because screenshots are not data."""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from slm.client import Ollama


@dataclass
class Sample:
    model: str
    prompt_id: str
    run: int
    ttft_ms: float
    tokens_per_sec: float
    total_ms: float
    output_tokens: int


def measure(client: Ollama, model: str, prompt_id: str, prompt: str, run: int) -> Sample:
    t0 = time.perf_counter()
    first = None
    tokens = 0
    for ev in client.stream(model, prompt, temperature=0.0):
        if ev.done:
            break
        if first is None:
            first = time.perf_counter()
        tokens += 1
    end = time.perf_counter()
    return Sample(
        model=model, prompt_id=prompt_id, run=run,
        ttft_ms=round((first - t0) * 1000, 1),
        tokens_per_sec=round(tokens / (end - t0), 2),
        total_ms=round((end - t0) * 1000, 1),
        output_tokens=tokens,
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
                    print(f"{model:<16} {p['id']:<8} run {run}  ttft {s.ttft_ms:>7.1f} ms  {s.tokens_per_sec:>6.1f} t/s")


if __name__ == "__main__":
    main()
