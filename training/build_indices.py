"""Build search indices from the rascenka catalogue.

Produces:
    * BM25 index (rank_bm25 ``BM25Okapi``) + obosn/naim list
    * FAISS IndexFlatIP over normalised SBERT embeddings + id map
    * SymSpell frequency dictionary (token -> token count)

For ``entity_type='rate'`` items, the indexed text is augmented with the
parent RateGroup name (and the group prefix when present), e.g.

    "Окраска поливинилацетатными … стен" + " | Малярные работы | Глава 15"

This is a soft form of two-stage search: a query like "малярные работы"
naturally pulls all rates from that group higher in the ranking,
without requiring a separate group-search service.
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from common.config import (
    BM25_INDEX_PATH,
    FAISS_ID_MAP_PATH,
    FAISS_INDEX_PATH,
    SBERT_BASE_MODEL,
    SBERT_EMBED_DIM,
    SBERT_MODEL_DIR,
    SYMSPELL_DICT_PATH,
)
from common.db import all_rascenki, all_rate_groups, rate_group_path
from common.logging_config import get_logger
from services.preprocessing.lemmatizer import lemmatize, tokenize

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Augmentation: rascenka name + parent group names
# ---------------------------------------------------------------------------

def _group_text_map() -> Dict[int, str]:
    """Build {group_id: 'Group | Parent group | …'} for all rate groups."""
    groups = {g["id"]: g for g in all_rate_groups()}
    out: Dict[int, str] = {}

    def _walk(gid: int) -> List[str]:
        chain: List[str] = []
        cur = gid
        for _ in range(10):
            g = groups.get(cur)
            if not g:
                break
            parts: List[str] = []
            if g.get("name_prefix"):
                parts.append(g["name_prefix"])
            if g.get("name"):
                parts.append(g["name"])
            chain.append(" ".join(parts).strip())
            if not g.get("parent_id"):
                break
            cur = g["parent_id"]
        return chain

    for gid in groups:
        out[gid] = " | ".join(_walk(gid))
    return out


def _augmented_text(rascenka: dict, group_text: Dict[int, str]) -> str:
    """Return text used both for BM25 tokenisation and SBERT encoding."""
    base = rascenka["naim"] or ""
    if rascenka.get("entity_type") == "rate" and rascenka.get("rate_group_id"):
        gtext = group_text.get(rascenka["rate_group_id"])
        if gtext:
            return f"{base} | {gtext}"
    return base


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def build_bm25() -> int:
    from rank_bm25 import BM25Okapi

    rows = all_rascenki()
    if not rows:
        log.warning("No rascenki – BM25 not built")
        return 0

    group_text = _group_text_map()

    obosn_list = [r["obosn"] for r in rows]
    naim_list = [r["naim"] for r in rows]
    entity_types = [r.get("entity_type", "rate") for r in rows]
    rate_group_ids = [r.get("rate_group_id") for r in rows]

    augmented = [_augmented_text(r, group_text) for r in rows]
    tokenised = [lemmatize(t) or tokenize(t) for t in augmented]

    index = BM25Okapi(tokenised)
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(
            {
                "index": index,
                "obosn_list": obosn_list,
                "naim_list": naim_list,
                "entity_types": entity_types,
                "rate_group_ids": rate_group_ids,
            },
            f,
        )
    log.info("Built BM25 index with %d docs -> %s", len(rows), BM25_INDEX_PATH)
    return len(rows)


# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------

def build_faiss() -> int:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning("faiss/sentence-transformers missing – FAISS not built")
        return 0

    rows = all_rascenki()
    if not rows:
        log.warning("No rascenki – FAISS not built")
        return 0

    group_text = _group_text_map()

    obosn_list = [r["obosn"] for r in rows]
    naim_list = [r["naim"] for r in rows]
    entity_types = [r.get("entity_type", "rate") for r in rows]
    rate_group_ids = [r.get("rate_group_id") for r in rows]
    augmented = [_augmented_text(r, group_text) for r in rows]

    model_to_load = str(SBERT_MODEL_DIR) if Path(SBERT_MODEL_DIR).exists() else SBERT_BASE_MODEL
    try:
        model = SentenceTransformer(model_to_load)
    except Exception as exc:
        log.warning("Could not load SBERT (%s) – FAISS not built", exc)
        return 0

    log.info("Encoding %d rascenki with %s", len(rows), model_to_load)
    embeddings = model.encode(
        augmented,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]

    # For small bases IndexFlatIP gives exact NN at acceptable cost;
    # switch to IndexIVFFlat for >100k vectors.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_ID_MAP_PATH, "wb") as f:
        pickle.dump(
            {
                "obosn_list": obosn_list,
                "naim_list": naim_list,
                "entity_types": entity_types,
                "rate_group_ids": rate_group_ids,
            },
            f,
        )
    log.info("Built FAISS index (dim=%d) -> %s", dim, FAISS_INDEX_PATH)
    return len(rows)


# ---------------------------------------------------------------------------
# SymSpell dictionary
# ---------------------------------------------------------------------------

def build_symspell_dict() -> int:
    rows = all_rascenki()
    if not rows:
        log.warning("No rascenki – SymSpell dict not built")
        return 0

    counter: Counter[str] = Counter()
    for r in rows:
        for tok in tokenize(r["naim"]):
            if tok.isalpha() and len(tok) >= 3:
                counter[tok.lower()] += 1

    # Also include rate-group names so the speller knows section/chapter words
    for g in all_rate_groups():
        for field in ("name", "name_prefix"):
            value = g.get(field)
            if not value:
                continue
            for tok in tokenize(value):
                if tok.isalpha() and len(tok) >= 3:
                    counter[tok.lower()] += 1

    SYMSPELL_DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SYMSPELL_DICT_PATH, "w", encoding="utf-8") as f:
        for term, cnt in counter.most_common():
            f.write(f"{term} {cnt}\n")
    log.info("Built SymSpell dictionary with %d terms -> %s",
             len(counter), SYMSPELL_DICT_PATH)
    return len(counter)


# ---------------------------------------------------------------------------
# All in one
# ---------------------------------------------------------------------------

def build_all(skip_faiss: bool = False) -> None:
    build_symspell_dict()
    build_bm25()
    if skip_faiss:
        log.info("Skipping FAISS index (--skip-faiss)")
    else:
        build_faiss()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-faiss", action="store_true", help="Skip SBERT/FAISS index build")
    args = p.parse_args()
    build_all(skip_faiss=args.skip_faiss)
