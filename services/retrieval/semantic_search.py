"""Semantic search using a fine-tuned SBERT model + FAISS.

The FAISS index and the (row_idx -> obosn) map are built once during
training and serialised to disk. At service start they are loaded into
memory.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from common.config import (
    FAISS_ID_MAP_PATH,
    FAISS_INDEX_PATH,
    SBERT_BASE_MODEL,
    SBERT_MODEL_DIR,
)
from common.logging_config import get_logger

log = get_logger(__name__)

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


class SemanticSearch:
    def __init__(
        self,
        index_path: Optional[Path] = None,
        id_map_path: Optional[Path] = None,
        model_dir: Optional[Path] = None,
    ):
        self.index = None
        self.obosn_list: List[str] = []
        self.naim_list: List[str] = []
        self.entity_types: List[str] = []
        self.rate_group_ids: List[Optional[int]] = []
        self.model = None

        if faiss is None or SentenceTransformer is None:
            log.warning("faiss or sentence-transformers missing – semantic disabled")
            return

        model_dir = model_dir or SBERT_MODEL_DIR
        model_to_load = str(model_dir) if Path(model_dir).exists() else SBERT_BASE_MODEL
        try:
            self.model = SentenceTransformer(model_to_load)
            log.info("Loaded SBERT model: %s", model_to_load)
        except Exception as exc:
            log.warning("Could not load SBERT model: %s", exc)
            return

        index_path = index_path or FAISS_INDEX_PATH
        id_map_path = id_map_path or FAISS_ID_MAP_PATH

        if not Path(index_path).exists() or not Path(id_map_path).exists():
            log.warning("FAISS index/id-map missing – run training")
            return

        try:
            self.index = faiss.read_index(str(index_path))
            with open(id_map_path, "rb") as f:
                payload = pickle.load(f)
            self.obosn_list = payload["obosn_list"]
            self.naim_list = payload["naim_list"]
            self.entity_types = payload.get(
                "entity_types", ["rate"] * len(self.obosn_list)
            )
            self.rate_group_ids = payload.get(
                "rate_group_ids", [None] * len(self.obosn_list)
            )
            log.info("Loaded FAISS index with %d vectors", self.index.ntotal)
        except Exception as exc:
            log.warning("Could not load FAISS index: %s", exc)
            self.index = None

    def search(
        self,
        query_text: str,
        top_k: int = 100,
        entity_filter: Optional[List[str]] = None,
        group_filter: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Return list of dicts with entity_type and rate_group_id included.

        Filtering is done *after* the FAISS NN search; we over-fetch when
        filters are supplied to compensate.
        """
        if self.index is None or self.model is None or not query_text:
            return []

        # Over-fetch when filtering to avoid an empty result set.
        fetch_k = top_k * 4 if (entity_filter or group_filter) else top_k

        vec = self.model.encode([query_text], normalize_embeddings=True)
        vec = np.asarray(vec, dtype="float32")
        scores, idxs = self.index.search(vec, fetch_k)

        out: List[Dict] = []
        ent_set = set(entity_filter) if entity_filter else None
        grp_set = set(group_filter) if group_filter else None

        rank = 0
        for i, s in zip(idxs[0], scores[0]):
            if i < 0:
                continue
            etype = self.entity_types[i]
            gid = self.rate_group_ids[i]
            if ent_set is not None and etype not in ent_set:
                continue
            if grp_set is not None and gid not in grp_set:
                continue
            rank += 1
            out.append({
                "obosn": self.obosn_list[i],
                "naim": self.naim_list[i],
                "score": float(s),
                "rank": rank,
                "entity_type": etype,
                "rate_group_id": gid,
            })
            if rank >= top_k:
                break
        return out
