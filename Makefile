MODELS ?=
RUNS ?= 5

.PHONY: install pull bench schema temperature report serve

install:
	python -m venv .venv && .venv/bin/pip install -e .

pull:
	python -c "import yaml;[print(m['tag']) for m in yaml.safe_load(open('models.yaml'))['models']]" | xargs -n1 ollama pull

bench:
	python -m bench.run --runs $(RUNS) $(if $(MODELS),--models $(MODELS))

schema:
	python -m bench.schema --runs $(RUNS) $(if $(MODELS),--models $(MODELS))

temperature:
	python -m bench.temperature --runs $(RUNS)

report:
	python -m bench.report | tee results/REPORT.md

serve:
	uvicorn slm.api:app --port 8100
