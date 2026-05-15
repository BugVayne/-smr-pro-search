"""Layer 2: hybrid candidate retrieval service.

Runs BM25 and SBERT in parallel and merges with RRF. Results carry
entity_type and (for rates) a resolved rate_group_name so the UI and
downstream layers can render hierarchy without re-querying the DB.

    POST /retrieve  RetrieveRequest -> RetrieveResponse
"""
from __future__ import annotations

from flask import Flask, jsonify, request

from common.config import PORTS, RETRIEVAL_TOP_K
from common.db import get_rate_groups
from common.logging_config import get_logger
from common.models import CandidateScore, RetrieveRequest, RetrieveResponse

from services.retrieval.bm25_search import BM25Search
from services.retrieval.rrf_fusion import rrf
from services.retrieval.semantic_search import SemanticSearch

log = get_logger("retrieval")
app = Flask(__name__)

_bm25 = BM25Search()
_sbert = SemanticSearch()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "retrieval"})


@app.post("/retrieve")
def retrieve():
    payload = RetrieveRequest(**request.get_json(force=True))
    top_k = payload.top_k or RETRIEVAL_TOP_K
    log.info("retrieve: %r (lemmas=%s)", payload.corrected, payload.lemmas)

    # BM25 over lemmas + expanded terms
    bm_tokens = list(payload.lemmas) + list(payload.expanded_terms)
    bm_results = _bm25.search(bm_tokens, top_k=top_k)

    # SBERT over the corrected user query
    sbert_results = _sbert.search(payload.corrected, top_k=top_k)

    fused = rrf(bm_results, sbert_results, top_n=top_k)

    # Resolve rate group names in one batch
    group_ids = [c["rate_group_id"] for c in fused if c.get("rate_group_id")]
    groups = get_rate_groups(list(set(group_ids))) if group_ids else {}

    candidates = []
    for c in fused:
        gid = c.get("rate_group_id")
        gname = None
        if gid and gid in groups:
            g = groups[gid]
            gname = " ".join(filter(None, [g.get("name_prefix"), g.get("name")])).strip()
        candidates.append(
            CandidateScore(
                obosn=c["obosn"],
                naim=c["naim"],
                entity_type=c.get("entity_type", "rate"),
                rate_group_id=gid,
                rate_group_name=gname,
                bm25_score=c["bm25_score"],
                sbert_score=c["sbert_score"],
                rrf_score=c["rrf_score"],
                bm25_rank=c["bm25_rank"],
                sbert_rank=c["sbert_rank"],
            )
        )

    return jsonify(RetrieveResponse(candidates=candidates).model_dump())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORTS["retrieval"], debug=False)
