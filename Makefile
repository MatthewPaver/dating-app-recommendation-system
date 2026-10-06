PYTHON ?= python3
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
PIP_BIN := $(PYTHON_BIN) -m pip
SAMPLE ?= examples/sample_swipes.csv
SYNTHETIC ?= data/synthetic_swipes.csv

.PHONY: venv install install-cli synthetic movielens summary demo test notebook

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP_BIN) install --upgrade pip
	$(PIP_BIN) install -r requirements.txt

install-cli: venv
	$(PIP_BIN) install -r requirements-cli.txt

synthetic:
	$(PYTHON_BIN) synthetic_swipes.py --output $(SYNTHETIC)

movielens: install
	$(PYTHON_BIN) scripts/download_movielens.py --accept-terms

summary: install-cli
	$(PYTHON_BIN) recommender.py --csv $(SAMPLE) summary

demo: install-cli
	$(PYTHON_BIN) recommender.py --csv $(SAMPLE) evaluate --top-k 2
	$(PYTHON_BIN) recommender.py --csv $(SAMPLE) recommend --user-id u1 --top-k 2

test: install-cli
	$(PYTHON_BIN) -m pytest tests -q

notebook: install synthetic
	$(PYTHON_BIN) -m jupyter lab recommendation_system_walkthrough.ipynb
