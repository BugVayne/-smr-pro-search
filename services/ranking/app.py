"""Layer 3: re-ranking service.

    POST /rank  RankRequest -> RankResponse
"""
from __future__ import annotations

from flask import Flask, jsonify, request

from common.config import PORTS, RANKING_TOP_K
from common.logging_config import get_logger
from common.models import RankedItem, RankRequest, RankResponse

from services.ranking.feature_extractor import FEATURE_NAMES, build_feature_matrix
from services.ranking.ranker import Ranker

log = get_logger("ranking")
app = Flask(__name__)

_ranker = Ranker()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ranking"})


@app.post("/rank")
def rank():
    payload = RankRequest(**request.get_json(force=True))

    if not payload.candidates:
        log.warning("RANK  → 0 candidates received, skipping")
        return jsonify(RankResponse(items=[]).model_dump())

    features = build_feature_matrix(payload.lemmas, payload.candidates, payload.context)
    scores = _ranker.score(features)

    scored = list(zip(payload.candidates, scores, features))
    scored.sort(key=lambda x: x[1], reverse=True)

    top_k = payload.top_k or RANKING_TOP_K
    items = [
        RankedItem(
            obosn=c.obosn,
            naim=c.naim,
            score=float(s),
            entity_type=c.entity_type,
            rate_group_id=c.rate_group_id,
            rate_group_name=c.rate_group_name,
            features=dict(zip(FEATURE_NAMES, [float(x) for x in feats])),
        )
        for c, s, feats in scored[:top_k]
    ]

    ranker_mode = "catboost" if _ranker._model is not None else "linear_fallback"
    top3_lines = ", ".join(
        f"{it.obosn}(score={it.score:.4f} bm25={it.features['f1_bm25']:.2f} "
        f"sbert={it.features['f2_sbert']:.3f} ovlp={it.features['f3_overlap']:.2f})"
        for it in items[:3]
    )
    log.info("RANK  «%s»", payload.query)
    log.info("  ranker    : %s", ranker_mode)
    log.info("  input     : %d candidates → top-%d", len(payload.candidates), top_k)
    log.info("  top3      : %s", top3_lines or "—")
    if payload.context:
        log.info("  context   : %s", payload.context)

    return jsonify(RankResponse(items=items).model_dump())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORTS["ranking"], debug=False)
