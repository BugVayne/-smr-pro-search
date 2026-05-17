"""Pipeline orchestrator.

Calls the four downstream microservices in order:

    preprocess  →  retrieve  →  rank  →  postprocess

Records per-step latency. All HTTP calls share a single Session for
connection pooling.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import requests

from common.config import (
    POSTPROCESS_URL,
    PREPROCESS_URL,
    RANKING_URL,
    RETRIEVAL_URL,
)
from common.db import get_rascenka
from common.logging_config import get_logger
from common.models import (
    PostprocessRequest,
    PostprocessResponse,
    PreprocessRequest,
    PreprocessResponse,
    RankRequest,
    RankResponse,
    RankedItem,
    RetrieveRequest,
    RetrieveResponse,
    SearchRequest,
    SearchResponse,
)
from services.postprocessing.kit_builder import enrich_with_related

log = get_logger("gateway.pipeline")
_session = requests.Session()

_RETRY_DELAYS = [1.0, 2.0, 4.0]  # seconds between retries on connection errors


def _post(url: str, payload: Dict[str, Any], timeout: float = 30.0):
    last_exc: Exception = RuntimeError("no attempts")
    for attempt, delay in enumerate([0.0] + _RETRY_DELAYS):
        if delay:
            log.warning("Retrying %s (attempt %d) in %.0fs…", url, attempt + 1, delay)
            time.sleep(delay)
        try:
            r = _session.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError as exc:
            last_exc = exc
        except requests.HTTPError:
            raise
    raise last_exc


def run_pipeline(req: SearchRequest) -> SearchResponse:
    timings: Dict[str, float] = {}

    # 1. preprocess
    t = time.perf_counter()
    pre_raw = _post(
        f"{PREPROCESS_URL}/preprocess",
        PreprocessRequest(query=req.query).model_dump(),
    )
    pre = PreprocessResponse(**pre_raw)
    timings["preprocess"] = (time.perf_counter() - t) * 1000

    # Short-circuit: if query is a direct rate code, look it up in DB and return.
    if pre.extracted_code:
        row = get_rascenka(pre.extracted_code)
        if row:
            item = RankedItem(
                obosn=row["obosn"],
                naim=row["naim"],
                score=1.0,
                entity_type=row.get("entity_type", "rate"),
                rate_group_id=row.get("rate_group_id"),
            )
            items = enrich_with_related([item])
            return SearchResponse(
                query=req.query,
                corrected=pre.corrected,
                query_type="atomic",
                items=items,
                timings_ms=timings,
                debug={
                    "preprocess": {
                        "original": pre.original,
                        "corrected": pre.corrected,
                        "lemmas": pre.lemmas,
                        "expanded_terms": pre.expanded_terms,
                        "query_type": pre.query_type,
                        "extracted_code": pre.extracted_code,
                    },
                    "direct_lookup": {"obosn": pre.extracted_code, "found": True},
                },
            )
        # Code not found in DB — fall through to full pipeline

    # 2. retrieve
    t = time.perf_counter()
    ret_raw = _post(
        f"{RETRIEVAL_URL}/retrieve",
        RetrieveRequest(
            corrected=pre.corrected,
            lemmas=pre.lemmas,
            expanded_terms=pre.expanded_terms,
        ).model_dump(),
    )
    ret = RetrieveResponse(**ret_raw)
    timings["retrieve"] = (time.perf_counter() - t) * 1000

    # 3. rank
    t = time.perf_counter()
    rank_raw = _post(
        f"{RANKING_URL}/rank",
        RankRequest(
            query=pre.corrected,
            lemmas=pre.lemmas,
            candidates=ret.candidates,
            context=req.context,
        ).model_dump(),
    )
    rank = RankResponse(**rank_raw)
    timings["rank"] = (time.perf_counter() - t) * 1000

    # 4. postprocess
    t = time.perf_counter()
    post_raw = _post(
        f"{POSTPROCESS_URL}/postprocess",
        PostprocessRequest(
            items=rank.items,
            query_type=pre.query_type,
            matched_ptms=ret.matched_ptms,
        ).model_dump(),
    )
    post = PostprocessResponse(**post_raw)
    timings["postprocess"] = (time.perf_counter() - t) * 1000

    return SearchResponse(
        query=req.query,
        corrected=pre.corrected,
        query_type=pre.query_type,
        items=post.items,
        kits=post.kits,
        ptm_groups=post.ptm_groups,
        timings_ms=timings,
        debug={
            "preprocess": {
                "original": pre.original,
                "corrected": pre.corrected,
                "lemmas": pre.lemmas,
                "expanded_terms": pre.expanded_terms,
                "query_type": pre.query_type,
            },
            "retrieve": {
                "candidates_count": len(ret.candidates),
                "matched_ptms": [
                    {"ptm_naim": p.ptm_naim, "overlap": round(p.overlap_score, 3)}
                    for p in ret.matched_ptms
                ],
                "top_candidates": [
                    {
                        "obosn": c.obosn,
                        "naim": c.naim,
                        "bm25": round(c.bm25_score, 4),
                        "sbert": round(c.sbert_score, 4),
                        "rrf": round(c.rrf_score, 4),
                    }
                    for c in ret.candidates[:5]
                ],
            },
            "rank": {
                "items_count": len(rank.items),
                "top_items": [
                    {
                        "obosn": i.obosn,
                        "naim": i.naim,
                        "score": round(i.score, 4),
                        "features": {k: round(v, 4) for k, v in i.features.items()},
                    }
                    for i in rank.items[:5]
                ],
            },
            "postprocess": {
                "items_count": len(post.items),
                "kits_count": len(post.kits),
                "ptm_groups_count": len(post.ptm_groups),
                "items_with_related": sum(1 for it in post.items if it.related),
            },
        },
    )
