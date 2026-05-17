"""CatBoost-based pointwise Learning-to-Rank.

When the trained CatBoost model is unavailable (e.g. before the first
training run) the ranker falls back to a heuristic linear combination
of features so the rest of the pipeline still works end-to-end.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from common.config import CATBOOST_MODEL_PATH
from common.logging_config import get_logger

log = get_logger(__name__)


class Ranker:
    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        model_path = model_path or CATBOOST_MODEL_PATH

        if not Path(model_path).exists():
            log.warning("CatBoost model %s not found – using linear fallback", model_path)
            return

        try:
            from catboost import CatBoost
            m = CatBoost()
            m.load_model(str(model_path))
            self._model = m
            log.info("Loaded CatBoost ranker from %s", model_path)
        except Exception as exc:
            log.warning("Could not load CatBoost: %s", exc)

    def score(self, features: List[List[float]]) -> List[float]:
        if not features:
            return []
        if self._model is not None:
            try:
                from catboost import Pool
                pool = Pool(data=features)
                preds = self._model.predict(pool)
                return [float(p) for p in preds]
            except Exception as exc:
                log.warning("CatBoost prediction failed (%s) – using fallback", exc)

        return [self._fallback_score(f) for f in features]

    @staticmethod
    def _fallback_score(f: List[float]) -> float:
        """Hand-tuned linear combination – mirrors LTR weights.

        Heavy weight on RRF-style signals (f4, f5) and SBERT (f2), with a
        smaller contribution from historical priors (f6, f7) and PTM (f12).
        f13 (numeric match) acts as a multiplier-like signal: 1.0 when no
        numbers in query (neutral), <1.0 when number is absent from candidate.
        """
        # Pad to expected length (13 features)
        f = list(f) + [1.0] * max(0, 13 - len(f))  # f13 defaults to 1.0 (neutral)
        return (
            0.23 * f[1]    # f2  sbert
            + 0.12 * f[0]  # f1  bm25
            + 0.15 * f[3]  # f4  inv bm25 rank
            + 0.15 * f[4]  # f5  inv sbert rank
            + 0.15 * f[12] # f13 numeric match  ← new
            + 0.08 * f[11] # f12 ptm score
            + 0.05 * f[2]  # f3  overlap
            + 0.04 * f[5]  # f6  prior
            + 0.03 * f[6]  # f7  context
        )
