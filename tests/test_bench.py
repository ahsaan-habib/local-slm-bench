import csv
import sys

import pytest

from bench import common, report, run, temperature
from bench.rss import RSSWatcher


def test_models_and_suite_files_are_well_formed():
    for set_name in ("compare", "quant"):
        models = common.load_models(set_name=set_name)
        assert models and all({"tag", "label", "quant"} <= m.keys() for m in models)
    first = common.load_models()[0]["tag"]
    assert [m["tag"] for m in common.load_models(only=[first])] == [first]
    suite = common.load_suite()
    assert len(suite) == 30 and len({r["id"] for r in suite}) == 30
    assert all(r["text"] in r["prompt"] for r in common.load_suite(kind="extract"))


def test_measure_separates_ttft_from_decode(mock_ollama, monkeypatch):
    monkeypatch.setattr(run, "RSSWatcher", lambda: type("W", (), {"start": lambda s: s, "stop": lambda s: 512.0})())
    monkeypatch.setattr(run, "loaded_model", lambda m: {"vram_mb": 3000, "fully_in_vram": True})
    s = run.measure(mock_ollama(["a", "b", "c", "d"]), "m", "p1", "prompt", 0)
    assert s.output_tokens == 4 and s.ttft_ms >= 0 and s.decode_tps > 0
    assert s.server_decode_tps == 4.0 and s.server_prefill_ms == 5.0 and s.peak_rss_mb == 512.0


def test_measure_with_no_tokens_does_not_crash(mock_ollama, monkeypatch):
    monkeypatch.setattr(run, "RSSWatcher", lambda: type("W", (), {"start": lambda s: s, "stop": lambda s: 0.0})())
    monkeypatch.setattr(run, "loaded_model", lambda m: {"vram_mb": 0, "fully_in_vram": False})
    s = run.measure(mock_ollama([]), "m", "p1", "prompt", 0)
    assert s.output_tokens == 0 and s.decode_tps == 0.0


def test_rss_watcher_samples_until_stopped(monkeypatch):
    vals = iter([100, 300 * 2**20, 200])
    monkeypatch.setattr("bench.rss._ollama_rss", lambda: next(vals, 0))
    w = RSSWatcher(interval=0.001).start()
    import time
    time.sleep(0.05)
    assert w.stop() == 300.0


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_report_charges_invalid_json_against_throughput(tmp_path, monkeypatch, capsys):
    b = [{"label": "Qwen3 4B", "quant": "Q4_K_M", "cold_load_ms": "2400", "model": "q", "decode_tps": tps,
          "ttft_ms": "300", "peak_rss_mb": "3000", "fully_in_vram": "True"} for tps in ("40", "44", "42")]
    sch = [{"model": "q", "valid_first": v} for v in ("True", "True", "True", "False")]
    write_csv(tmp_path / "bench.csv", b)
    write_csv(tmp_path / "schema.csv", sch)
    monkeypatch.setattr(sys, "argv", ["report", "--bench", str(tmp_path / "bench.csv"),
                                      "--schema", str(tmp_path / "schema.csv")])
    report.main()
    row = [l for l in capsys.readouterr().out.splitlines() if l.startswith("| Qwen3")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3] == "yes" and cells[4] == "2.4 s" and cells[6] == "42.0"
    assert cells[7] == "75.0%" and cells[8] == "33.6"       # 42 / (1 + 0.25)


def test_report_without_results_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["report"])
    with pytest.raises(SystemExit, match="make bench"):
        report.main()


def test_shape_compares_json_structure():
    assert temperature._shape('{"b": 1, "a": 2}') == "a,b"
    assert temperature._shape("[1]") == "list" and temperature._shape("nope") == "<invalid>"
