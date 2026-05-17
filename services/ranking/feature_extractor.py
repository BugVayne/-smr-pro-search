"""Feature extraction for the ranker.

Produces the 13-feature vector:

    f1  – BM25 score                        (from retrieval)
    f2  – SBERT cosine similarity           (from retrieval)
    f3  – token overlap fraction
    f4  – inverse BM25 rank
    f5  – inverse SBERT rank
    f6  – normalised absolute frequency     (from KB profiles)
    f7  – contextual frequency              (from KB profiles)
    f8  – is-current (always 1.0 for now)
    f9  – object type (categorical, hashed)
    f10 – region (categorical, hashed)
    f11 – glava (categorical, hashed)
    f12 – PTM aggregated score              (from PTM retrieval branch)
    f13 – numeric token match fraction      (size/dimension specificity)
"""
from __future__ import annotations

import re
from typing import Dict, List

from common.db import get_profiles
from common.models import CandidateScore

_HAS_DIGIT = re.compile(r'\d')
_DIGITS_ONLY = re.compile(r'\d+')


def _token_overlap(lemmas: List[str], naim: str) -> float:
    if not lemmas:
        return 0.0
    naim_lower = naim.lower()
    hits = sum(1 for t in lemmas if t in naim_lower)
    return hits / len(lemmas)


def _numeric_match(lemmas: List[str], naim: str) -> float:
    """Fraction of digit-containing query tokens found in candidate name.

    Returns 1.0 when the query has no numeric tokens (neutral, no penalty).
    Uses digit-boundary regex so "50" matches "ду50" but not "150" or "500".
    """
    numeric = [t for t in lemmas if _HAS_DIGIT.search(t)]
    if not numeric:
        return 1.0
    naim_lower = naim.lower()
    hits = 0
    for tok in numeric:
        if tok in naim_lower:  # exact token substring (e.g. "ду50" in "ду50")
            hits += 1
            continue
        # Also try matching just the digit run with boundary guard
        # so "50" from query matches "ду50" in name but not "150"
        m = _DIGITS_ONLY.search(tok)
        if m and re.search(r'(?<!\d)' + m.group(0) + r'(?!\d)', naim_lower):
            hits += 1
    return hits / len(numeric)


def _cat_hash(value: str, modulo: int = 100) -> float:
    return float(hash(value) % modulo) / modulo if value else 0.0


def build_feature_matrix(
    lemmas: List[str],
    candidates: List[CandidateScore],
    context: Dict[str, str],
) -> List[List[float]]:
    """Build a list of per-candidate feature vectors."""
    obosn_list = [c.obosn for c in candidates]
    profiles = get_profiles(obosn_list)

    ptm_kod = context.get("ptm_kod", "")
    glava = context.get("glava", "")
    object_type = context.get("object_type", "")
    region = context.get("region", "")

    matrix: List[List[float]] = []
    for cand in candidates:
        prof = profiles.get(cand.obosn, {})
        f6 = float(prof.get("f_norm", 0.0))
        # contextual frequency: how often this rascenka appears in PTMs
        # with the same ptm_kod as the current context
        ctx_map = prof.get("ctx", {})
        f7 = float(ctx_map.get(ptm_kod, 0)) / max(prof.get("f_abs", 1), 1)

        feats = [
            cand.bm25_score,                                     # f1
            cand.sbert_score,                                     # f2
            _token_overlap(lemmas, cand.naim),                    # f3
            1.0 / cand.bm25_rank  if cand.bm25_rank  else 0.0,  # f4
            1.0 / cand.sbert_rank if cand.sbert_rank else 0.0,  # f5
            f6,                                                   # f6
            f7,                                                   # f7
            1.0,                                                  # f8 (always current)
            _cat_hash(object_type),                               # f9
            _cat_hash(region),                                    # f10
            _cat_hash(glava),                                     # f11
            cand.ptm_score,                                       # f12
            _numeric_match(lemmas, cand.naim),                    # f13
        ]
        matrix.append(feats)
    return matrix


FEATURE_NAMES = [
    "f1_bm25", "f2_sbert", "f3_overlap",
    "f4_inv_bm25_rank", "f5_inv_sbert_rank",
    "f6_f_norm", "f7_ctx", "f8_actual",
    "f9_object_type", "f10_region", "f11_glava",
    "f12_ptm", "f13_numeric_match",
]
