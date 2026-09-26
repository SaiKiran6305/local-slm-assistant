.PHONY: install test bench bench-grammar serve serve-ollama docker

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -q

bench:
	python -m bench.run --provider simulated --out bench_report.json

bench-grammar:
	python -m bench.run --provider simulated --grammar-first --out bench_grammar.json

# Real numbers. Needs `ollama serve` and at least one pulled model.
bench-real:
	python -m bench.run --provider ollama --out bench_real.json

serve:
	uvicorn app.main:app --reload --port 8000

serve-ollama:
	PROVIDER=ollama LOCAL_MODEL=llama3.2:3b uvicorn app.main:app --reload --port 8000

docker:
	docker build -t local-slm-assistant . && docker run -p 8000:8000 local-slm-assistant
