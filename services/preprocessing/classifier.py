"""Query type classifier.

Determines whether the user query is *atomic* (a specific rascenka or
single concrete operation) or *aggregated* (a high-level concept like
"тёплый пол" that should expand into a kit of rascenki).

Priority chain:
  1. Regex fast-path  – code patterns (e.g. "ЕР46-01-008") -> always atomic
  2. ML model         – TF-IDF + LogisticRegression trained by
                        training.train_query_classifier on PTM names
                        (aggregated) vs rascenka name fragments (atomic)
  3. Length heuristic – fallback when model is absent:
                        1-2 words -> atomic, 3+ words -> aggregated
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

from common.config import QTYPE_MODEL_PATH
from common.logging_config import get_logger

log = get_logger(__name__)

CODE_RE = re.compile(r"[А-ЯA-Z]+[0-9][0-9А-ЯA-Z\-]+")


class QueryClassifier:
    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        path = Path(model_path) if model_path else QTYPE_MODEL_PATH
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self._model = pickle.load(f)
                log.info("Loaded query-type classifier from %s", path)
            except Exception as exc:
                log.warning("Could not load query classifier: %s – using heuristic", exc)
        else:
            log.warning("Query-type classifier not found at %s – using length heuristic", path)

    def predict(self, query: str) -> str:
        q = query.strip().lower()

        # 1. Code pattern -> always atomic
        if CODE_RE.search(query):
            return "atomic"

        # 2. ML model (threshold 0.42: slightly favour aggregated since
        #    missing a kit hurts more than an unnecessary kit suggestion)
        if self._model is not None:
            try:
                proba = self._model.predict_proba([q])[0]
                classes = self._model.classes_
                scores = dict(zip(classes, proba))
                return "aggregated" if scores.get("aggregated", 0) >= 0.42 else "atomic"
            except Exception as exc:
                log.warning("Classifier prediction failed: %s – falling back to heuristic", exc)

        # 3. Length heuristic fallback: short query = general concept, long = specific rate
        return "aggregated" if len(q.split()) <= 2 else "atomic"
