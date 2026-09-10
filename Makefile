setup:
	python -m pip install -r requirements-dev.txt

download:
	python -m scripts.download_data

train:
	python -m src.train

evaluate:
	python -m src.evaluate

serve:
	uvicorn src.banking_router.api.app:app --host 0.0.0.0 --port 8000

test:
	python -m pytest -q

lint:
	python -m ruff check src scripts tests
	python -m ruff format --check src scripts tests
