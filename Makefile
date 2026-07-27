PYTHON ?= python3
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
PIP_BIN := $(PYTHON_BIN) -m pip
SAMPLE ?= examples/sample_swipes.csv
SYNTHETIC ?= data/synthetic_swipes.csv

.PHONY: venv install synthetic summary demo test notebook

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP_BIN) install --upgrade pip
	$(PIP_BIN) install -r requirements.txt

synthetic:
	$(PYTHON_BIN) synthetic_swipes.py --output $(SYNTHETIC)

summary: install
	$(PYTHON_BIN) recommender.py --csv $(SAMPLE) summary

demo: install
	$(PYTHON_BIN) recommender.py --csv $(SAMPLE) evaluate --top-k 2
	$(PYTHON_BIN) recommender.py --csv $(SAMPLE) recommend --user-id u1 --top-k 2

test: install
	$(PYTHON_BIN) -m pytest tests -q

notebook: install synthetic
	$(PYTHON_BIN) -m jupyter lab recommendation_system_walkthrough.ipynb
