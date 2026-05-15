"""Reciprocal Rank Fusion (RRF).

Merges two ranked lists by summing ``1/(k + rank)`` across runs.
Inputs are lists of result dicts; the union preserves entity_type and
rate_group_id metadata from whichever branch produced the document.
"""
from __future__ import annotations

from typing import Dict, List

from common.config import RRF_K


def rrf(
    bm25_results: List[Dict],
    sbert_results: List[Dict],
    k: int = RRF_K,
    top_n: int = 100,
) -> List[Dict]:
    """Combine two result lists.

    Each input element is a dict with at least:
        obosn, naim, score, rank, entity_type, rate_group_id

    Returns a list of dicts with the unified RRF score plus the
    underlying signals from each branch.
    """
    scores: Dict[str, float] = {}
    bm25_map: Dict[str, Dict] = {r["obosn"]: r for r in bm25_results}
    sbert_map: Dict[str, Dict] = {r["obosn"]: r for r in sbert_results}

    for r in bm25_results:
        scores[r["obosn"]] = scores.get(r["obosn"], 0.0) + 1.0 / (k + r["rank"])
    for r in sbert_results:
        scores[r["obosn"]] = scores.get(r["obosn"], 0.0) + 1.0 / (k + r["rank"])

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    out: List[Dict] = []
    for obosn, rrf_score in fused:
        b = bm25_map.get(obosn)
        s = sbert_map.get(obosn)
        any_row = b or s  # at least one is non-null
        out.append({
            "obosn": obosn,
            "naim": any_row["naim"],
            "entity_type": any_row.get("entity_type", "rate"),
            "rate_group_id": any_row.get("rate_group_id"),
            "bm25_score": b["score"] if b else 0.0,
            "sbert_score": s["score"] if s else 0.0,
            "bm25_rank": b["rank"] if b else None,
            "sbert_rank": s["rank"] if s else None,
            "rrf_score": rrf_score,
        })
    return out
