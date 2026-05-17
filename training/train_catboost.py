"""Train a pointwise CatBoost ranker from user feedback.

Implicit-feedback labelling (mirrors §5.7 of the diploma):

    label 2 — rascenka selected by the user
    label 1 — rascenka edited/added after the search
    label 0 — shown but ignored

User feedback is stored in the ``user_feedback`` table by the gateway
``/feedback`` endpoint.
"""
from __future__ import annotations

import random
from typing import List, Tuple

from common.config import CATBOOST_MODEL_PATH
from common.db import all_feedback, all_rascenki
from common.logging_config import get_logger
from common.models import CandidateScore
from services.preprocessing.lemmatizer import lemmatize
from services.ranking.feature_extractor import FEATURE_NAMES, build_feature_matrix

log = get_logger(__name__)


def _build_training_set(min_examples: int = 20) -> Tuple[List[List[float]], List[int]]:
    feedback = all_feedback()
    if len(feedback) < min_examples:
        log.warning("Only %d feedback rows – CatBoost training skipped", len(feedback))
        return [], []

    # We need fabricated retrieval features for each feedback row. Without
    # online retrieval logs we approximate by recomputing rough text
    # features via the rascenka catalogue.
    rascenki = {r["obosn"]: r for r in all_rascenki()}
    X: List[List[float]] = []
    y: List[int] = []
    for fb in feedback:
        r = rascenki.get(fb["obosn"])
        if not r:
            continue
        cand = CandidateScore(
            obosn=fb["obosn"],
            naim=r["naim"],
            bm25_score=1.0,
            sbert_score=0.5,
            rrf_score=0.0,
            bm25_rank=1,
            sbert_rank=1,
        )
        feats = build_feature_matrix(
            lemmas=lemmatize(fb["query"]),
            candidates=[cand],
            context={},
        )
        X.append(feats[0])
        y.append(int(fb["label"]))

    return X, y


def train() -> bool:
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        log.warning("catboost not installed – skipping training")
        return False

    X, y = _build_training_set()
    if not X:
        return False

    try:
        import torch
        task_type = "GPU" if torch.cuda.is_available() else "CPU"
    except ImportError:
        task_type = "CPU"

    log.info("Training CatBoost on %d examples (task_type=%s)", len(X), task_type)
    model = CatBoostRegressor(
        iterations=200,
        depth=4,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type=task_type,
        verbose=False,
        feature_names=FEATURE_NAMES,
    )
    model.fit(X, y)
    CATBOOST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(CATBOOST_MODEL_PATH))
    log.info("Saved CatBoost ranker to %s", CATBOOST_MODEL_PATH)
    return True


if __name__ == "__main__":
    train()
