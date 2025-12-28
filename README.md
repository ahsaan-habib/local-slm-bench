# local-slm-bench

Can a 3–4B model running on one laptop do the job? This repo measures it
instead of arguing about it: same machine, same 30 prompts, five runs each,
everything written to CSV.

Default model is **Qwen3 4B** via [Ollama](https://ollama.com); Llama 3.2 3B
and Phi-4-mini are in `models.yaml` as the comparison set. Add anything Ollama
can pull.

## What it measures

| | How | Why |
|---|---|---|
| **TTFT** | client clock, request → first streamed token | what a person waiting feels |
| **Decode t/s** | tokens after the first ÷ time after the first | *not* tokens ÷ total — that folds prefill in and punishes long prompts |
| Ollama prefill / decode | `prompt_eval_duration`, `eval_count / eval_duration` | server-side cross-check of the above |
| **Peak RSS** | sampled RSS of the ollama runner processes | the model's memory, not this script's |
| VRAM residency | `/api/ps` `size_vram` vs `size` | a model that spills to RAM is a different model |
| Cold load | unload, then time an empty generate | matters for anything that scales to zero |
| **Schema compliance** | Pydantic validation, first try and after one repair | for extraction this decides, not speed |
| Effective t/s | decode t/s ÷ (1 + invalid rate) | charges each model for its retries |
| Temperature variance | distinct outputs / JSON shapes across runs at T=0 and T=0.7 | proves why structured work runs at 0 |

## Run it

```bash
make install && source .venv/bin/activate
make pull                 # ollama pull every tag in models.yaml
make bench                # -> results/bench-<host>-<time>.csv (+ .machine.json)
make schema               # -> results/schema-...csv
make temperature          # -> results/temperature-...csv
make report               # -> results/REPORT.md
```

`MODELS="qwen3:4b phi4-mini" make bench` for a subset, `RUNS=3` for a quick one.

Close other heavy apps, let the machine cool between runs, and keep it on AC
power. Every number is only valid for the machine in its `.machine.json`.

## Structured output

`slm/structured.py`: JSON mode → Pydantic → on failure, **one** repair attempt
with the validation error fed back → otherwise `None` and a warning log. Not a
`while True`. A model that failed twice with the error in front of it won't
succeed on the fifth try; it'll just add latency.

`bench.schema --constrained` runs the same thing with the JSON schema passed
to Ollama's constrained decoding, to see how much of the retry problem the
grammar removes for each model.

## API

```bash
make serve
curl -N localhost:8100/generate -d '{"prompt":"hi"}' -H 'content-type: application/json'
curl localhost:8100/extract -d '{"text":"move my 3pm with Sara to thursday"}' -H 'content-type: application/json'
```

`/extract` returns 422 with the raw model outputs when validation fails twice,
never a 200 with a guess.

## Caveats

One machine; thermals drift on long runs; 30 prompts separates big gaps, not
small ones; "quality" here is schema compliance, not whether the entities
were the right ones. Pin model tags and re-run rather than trusting any table,
including the one this produces.
