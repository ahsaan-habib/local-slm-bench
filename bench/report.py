"""Turn the latest CSVs into the decision table (markdown).

Medians, not means. Effective throughput charges each model for the retries
its invalid outputs cause: an invalid object costs another prefill + decode.

    python -m bench.report > results/REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def latest(kind: str) -> Path | None:
    files = sorted(Path("results").glob(f"{kind}-*.csv"))
    return files[-1] if files else None


def read(path: Path | None) -> list[dict]:
    return list(csv.DictReader(open(path))) if path else []


def med(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return statistics.median(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=None)
    ap.add_argument("--schema", type=Path, default=None)
    args = ap.parse_args()

    bench_file, schema_file = args.bench or latest("bench"), args.schema or latest("schema")
    bench, schema = read(bench_file), read(schema_file)
    if not bench:
        raise SystemExit("no results/bench-*.csv yet — run `make bench` first")

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in bench:
        by_model[r["model"]].append(r)
    invalid: dict[str, float] = {}
    for tag in by_model:
        rows = [r for r in schema if r["model"] == tag]
        if rows:
            invalid[tag] = sum(r["valid_first"] != "True" for r in rows) / len(rows)

    print(f"Source: `{bench_file}`" + (f", `{schema_file}`" if schema_file else ""))
    print()
    print("| Model | Quant | Peak RSS (MB) | In VRAM | Cold load | TTFT p50 | Decode t/s p50 "
          "| Valid JSON (1st try) | Effective t/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tag, rows in by_model.items():
        tps = med(rows, "decode_tps")
        inv = invalid.get(tag)
        eff = f"{tps / (1 + inv):.1f}" if inv is not None else "–"
        valid = f"{1 - inv:.1%}" if inv is not None else "–"
        in_vram = "yes" if all(r["fully_in_vram"] == "True" for r in rows) else "partial"
        print(f"| {rows[0]['label']} | {rows[0]['quant']} | {max(float(r['peak_rss_mb']) for r in rows):.0f} "
              f"| {in_vram} | {float(rows[0]['cold_load_ms']) / 1000:.1f} s | {med(rows, 'ttft_ms'):.0f} ms "
              f"| {tps:.1f} | {valid} | {eff} |")


if __name__ == "__main__":
    main()
