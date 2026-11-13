PY ?= python

.PHONY: install data train signals serve test lint

install:
	$(PY) -m pip install -r requirements-dev.txt

data:      ## download 2016+ quarters, load SQLite, run quality checks
	$(PY) -m etl.run --from-year 2016

train:
	$(PY) -m ml.train

signals:
	$(PY) -m stats.signals

serve:
	uvicorn app.main:app --reload --port 8000

lint:
	$(PY) -m pyflakes etl ml stats app tests

test:
	$(PY) -m pytest -q
