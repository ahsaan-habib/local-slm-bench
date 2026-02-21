"""Schema compliance per model: valid first time, valid after one repair,
and effective throughput once retries are paid for.

    python -m bench.schema --runs 5
"""
from __future__ import annotations

import argparse
import csv
import time

from slm.client import Ollama
from slm.structured import extract

from .common import load_models, load_suite, out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="compare", help="model set from models.yaml")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--constrained", action="store_true",
                    help="pass the JSON schema to ollama (grammar-constrained decoding)")
    args = ap.parse_args()

    client = Ollama()
    suite = load_suite(kind="extract")
    out = out_path("schema-constrained" if args.constrained else "schema")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "prompt_id", "run", "valid_first",
                                          "valid_after_retry", "attempts", "latency_ms"])
        w.writeheader()
        for m in load_models(only=args.models, set_name=args.set):
            first_ok = final_ok = n = 0
            for run in range(args.runs):
                for p in suite:
                    t0 = time.perf_counter()
                    a = extract(client, m["tag"], p["text"], schema=args.constrained)
                    ms = (time.perf_counter() - t0) * 1000
                    ok_first = a.attempts == 1 and a.result is not None
                    w.writerow({"model": m["tag"], "prompt_id": p["id"], "run": run,
                                "valid_first": ok_first, "valid_after_retry": a.result is not None,
                                "attempts": a.attempts, "latency_ms": round(ms, 1)})
                    first_ok += ok_first
                    final_ok += a.result is not None
                    n += 1
            print(f"{m['tag']:<16} valid first {first_ok / n:6.1%}   after one retry {final_ok / n:6.1%}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
