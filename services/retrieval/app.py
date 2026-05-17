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
from common.models import CandidateScore, PTMMatchInfo, Rascenka, RetrieveRequest, RetrieveResponse

from services.retrieval.bm25_search import BM25Search
from services.retrieval.ptm_search import PTMSearch
from services.retrieval.rrf_fusion import rrf
from services.retrieval.semantic_search import SemanticSearch

log = get_logger("retrieval")
app = Flask(__name__)

_bm25  = BM25Search()
_sbert = SemanticSearch()
_ptm   = PTMSearch()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "retrieval"})


def _fmt_top(results, score_key="score", n=3):
    return ", ".join(
        f"{r['obosn']}({r[score_key]:.3f})" for r in results[:n]
    ) or "—"


@app.post("/retrieve")
def retrieve():
    payload = RetrieveRequest(**request.get_json(force=True))
    top_k = payload.top_k or RETRIEVAL_TOP_K

    # BM25 over lemmas + expanded terms
    bm_tokens = list(payload.lemmas) + list(payload.expanded_terms)
    bm_results    = _bm25.search(bm_tokens, top_k=top_k)
    sbert_results = _sbert.search(payload.corrected, top_k=top_k)
    ptm_results   = _ptm.search(list(payload.lemmas), top_k=top_k)

    fused = rrf(bm_results, sbert_results, ptm_results, top_n=top_k)

    log.info("RETRIEVE «%s»", payload.corrected)
    log.info("  bm25 tokens(%d): %s", len(bm_tokens), bm_tokens)
    if bm_results:
        log.info("  bm25  -> %d hits  | top3: %s", len(bm_results), _fmt_top(bm_results))
    else:
        log.warning("  bm25  -> 0 hits (index missing or no match)")
    if sbert_results:
        log.info("  sbert -> %d hits  | top3: %s", len(sbert_results), _fmt_top(sbert_results))
    else:
        log.warning("  sbert -> 0 hits (index missing or model not loaded)")
    matched_ptms_raw = _ptm.get_matched_ptms(list(payload.lemmas))
    if ptm_results:
        matched_names = [p["ptm_naim"] for p in matched_ptms_raw] or _ptm.matched_ptm_names(list(payload.lemmas))
        log.info("  ptm   -> %d hits  | PTMs: %s | top3: %s",
                 len(ptm_results), matched_names, _fmt_top(ptm_results))
    else:
        log.info("  ptm   -> 0 hits (index missing or no PTM match)")
    log.info("  rrf   -> %d candidates | top3: %s",
             len(fused), _fmt_top(fused, score_key="rrf_score"))

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
                ptm_score=c["ptm_score"],
                rrf_score=c["rrf_score"],
                bm25_rank=c["bm25_rank"],
                sbert_rank=c["sbert_rank"],
                ptm_rank=c["ptm_rank"],
            )
        )

    matched_ptms = [
        PTMMatchInfo(
            ptm_naim=p["ptm_naim"],
            overlap_score=p["overlap_score"],
            top_rascenki=[
                Rascenka(
                    obosn=r["obosn"],
                    naim=r["naim"],
                    entity_type=r.get("entity_type", "rate"),
                    rate_group_id=r.get("rate_group_id"),
                )
                for r in p["top_rascenki"]
            ],
        )
        for p in matched_ptms_raw
    ]

    return jsonify(RetrieveResponse(candidates=candidates, matched_ptms=matched_ptms).model_dump())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORTS["retrieval"], debug=False)
