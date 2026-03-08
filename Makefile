MODELS ?=
RUNS ?= 5
SET ?= compare

.PHONY: install pull bench schema temperature report serve quant

install:
	python -m venv .venv && .venv/bin/pip install -e .

pull:
	python -c "import yaml;[print(m['tag']) for m in yaml.safe_load(open('models.yaml'))['sets']['$(SET)'] if 'local' not in m['tag']]" | xargs -n1 ollama pull

bench:
	python -m bench.run --set $(SET) --runs $(RUNS) $(if $(MODELS),--models $(MODELS))

schema:
	python -m bench.schema --set $(SET) --runs $(RUNS) $(if $(MODELS),--models $(MODELS))

temperature:
	python -m bench.temperature --runs $(RUNS)

report:
	python -m bench.report | tee results/REPORT.md

quant:
	./scripts/quantize.sh Qwen/Qwen3-4B-Instruct-2507 Q5_K_M qwen3-4b-instruct-local:q5_K_M
	$(MAKE) bench schema SET=quant

serve:
	uvicorn slm.api:app --port 8100
