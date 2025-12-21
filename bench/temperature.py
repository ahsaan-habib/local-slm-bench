"""Same prompts, N runs each, at two temperatures. Counts how many prompts
produced *any* variation across runs — and, for extraction, whether the JSON
structure itself changed (keys present/absent), not just the wording.

    python -m bench.temperature --model qwen3:4b --runs 5
"""
from __future__ import annotations

import argparse
import csv
import json

from slm.client import Ollama
from slm.structured import EXTRACT_PROMPT

from .common import load_suite, out_path


def _shape(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return "<invalid>"
    return ",".join(sorted(obj)) if isinstance(obj, dict) else type(obj).__name__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.7])
    args = ap.parse_args()

    client = Ollama()
    suite = load_suite()
    out = out_path("temperature")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "temperature", "prompt_id", "kind",
                                          "distinct_outputs", "distinct_shapes"])
        w.writeheader()
        for temp in args.temps:
            varied = shape_changed = 0
            for p in suite:
                is_extract = p["kind"] == "extract"
                prompt = EXTRACT_PROMPT.format(text=p["text"]) if is_extract else p["prompt"]
                outs = [client.generate(args.model, prompt, temperature=temp,
                                        fmt="json" if is_extract else None, num_predict=256)[0]
                        for _ in range(args.runs)]
                distinct = len(set(outs))
                shapes = len({_shape(o) for o in outs}) if is_extract else 1
                varied += distinct > 1
                shape_changed += shapes > 1
                w.writerow({"model": args.model, "temperature": temp, "prompt_id": p["id"],
                            "kind": p["kind"], "distinct_outputs": distinct, "distinct_shapes": shapes})
            print(f"T={temp}: {varied}/{len(suite)} prompts varied across {args.runs} runs, "
                  f"{shape_changed} changed JSON structure")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
