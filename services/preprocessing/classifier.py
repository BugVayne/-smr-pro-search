"""Query type classifier.

Determines whether the user query is *atomic* (a specific rascenka, a
code, a single concrete operation) or *aggregated* (a high-level concept
like "тёплый пол" that should expand into a kit of rascenki).

Combines two signals:

* a regex-based fast path for codes – these are always atomic;
* a small logistic-regression model on TF-IDF features trained on
  labelled examples (in ``data/processed/qtype_train.csv``).

If the trained model file does not exist we fall back to a simple
heuristic based on a list of known aggregated concepts.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

from common.config import QTYPE_MODEL_PATH
from common.logging_config import get_logger

log = get_logger(__name__)

CODE_RE = re.compile(r"[А-ЯA-Z]{2,}[0-9][0-9А-ЯA-Z\-]+")

AGGREGATED_CONCEPTS = {
    "теплый пол", "тёплый пол", "кровельный пирог", "кровельный",
    "вентилируемый фасад", "система отопления", "котельная",
    "напольное покрытие", "система кондиционирования",
}


class QueryClassifier:
    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        model_path = model_path or QTYPE_MODEL_PATH
        if Path(model_path).exists():
            try:
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                log.info("Loaded query-type classifier from %s", model_path)
            except Exception as exc:
                log.warning("Could not load query classifier: %s", exc)

    def predict(self, query: str) -> str:
        q = query.strip().lower()

        # Fast path: codes are always atomic
        if CODE_RE.search(query):
            return "atomic"

        # Heuristic: known aggregated concepts
        for concept in AGGREGATED_CONCEPTS:
            if concept in q:
                return "aggregated"

        if self._model is not None:
            try:
                return str(self._model.predict([q])[0])
            except Exception as exc:
                log.warning("Classifier prediction failed: %s", exc)

        # Default: short queries (1–2 words) usually atomic, longer
        # descriptive phrases aggregated.
        return "atomic" if len(q.split()) <= 2 else "aggregated"
