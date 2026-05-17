# Convenience targets for the SMR-Pro search project.

PY ?= python
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

.PHONY: help install init import sample train run stop test clean

help:
	@echo "Available targets:"
	@echo "  install      – install Python dependencies"
	@echo "  init         – create SQLite schema"
	@echo "  import       – import SMR-Pro catalogue from data/raw/smr_catalog.db"
	@echo "  sample       – load tiny synthetic demo data into the DB"
	@echo "  train        – run full training pipeline (uses your data)"
	@echo "  train-fast   – training without SBERT/CatBoost (dev cycle)"
	@echo "  run          – start all microservices + UI"
	@echo "  stop         – stop running services"
	@echo "  test         – run smoke test against running services"
	@echo "  clean        – remove our DB, indices and models"

install:
	$(PY) -m pip install -r requirements-dev.txt

init:
	$(PY) scripts/init_db.py

# Import the catalogue from your SMR-Pro SQLite file.
# Place it at data/raw/smr_catalog.db, or pass DB=path/to/your.db
DB ?= data/raw/smr_catalog.db
import: init
	$(PY) -m scripts.import_smr_db $(DB)

sample: init
	$(PY) scripts/load_sample_data.py

migrate-codes:
	$(PY) scripts/migrate_code_mapping.py

train:
	$(PY) -m training.pipeline

train-fast:
	$(PY) -m training.pipeline --skip-sbert --skip-catboost

train-bm25:
	$(PY) -m training.pipeline --skip-sbert --skip-catboost --skip-faiss

train-classifier:
	$(PY) -m training.train_query_classifier

run:
	$(PY) scripts/run_all.py

stop:
	$(PY) scripts/run_all.py stop

test:
	$(PY) scripts/smoke_test.py

clean: stop
	$(PY) scripts/clean.py
