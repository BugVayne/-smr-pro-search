"""BM25 search.

Uses the lightweight ``rank_bm25`` package instead of Elasticsearch –
for the project scale (tens of thousands of rascenki) this is fast
enough and avoids running a separate service. The index is built once
by ``training.build_indices`` and serialised to disk.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

from common.config import BM25_INDEX_PATH
from common.logging_config import get_logger

_NUMERIC_BOOST = 3  # repeat numeric tokens N times so BM25 weights them higher
_HAS_DIGIT = re.compile(r'\d')

log = get_logger(__name__)


class BM25Search:
    def __init__(self, index_path: Optional[Path] = None):
        self.index = None
        self.obosn_list: List[str] = []
        self.naim_list: List[str] = []
        self.entity_types: List[str] = []
        self.rate_group_ids: List[Optional[int]] = []
        index_path = index_path or BM25_INDEX_PATH

        if not Path(index_path).exists():
            log.warning("BM25 index %s not found – run training", index_path)
            return

        try:
            with open(index_path, "rb") as f:
                payload = pickle.load(f)
            self.index = payload["index"]
            self.obosn_list = payload["obosn_list"]
            self.naim_list = payload["naim_list"]
            # Newer fields – default if missing (older indices)
            self.entity_types = payload.get(
                "entity_types", ["rate"] * len(self.obosn_list)
            )
            self.rate_group_ids = payload.get(
                "rate_group_ids", [None] * len(self.obosn_list)
            )
            log.info("Loaded BM25 index with %d docs", len(self.obosn_list))
        except Exception as exc:
            log.warning("Could not load BM25 index: %s", exc)

    def search(
        self,
        query_tokens: List[str],
        top_k: int = 100,
        entity_filter: Optional[List[str]] = None,
        group_filter: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Return list of dicts: {obosn, naim, score, rank, entity_type, rate_group_id}.

        ``entity_filter`` and ``group_filter`` are optional whitelists.
        If omitted, all entity types and groups are eligible.
        """
        if self.index is None or not query_tokens:
            return []

        # Repeat numeric tokens so BM25 weights them above low-IDF threshold.
        # rank_bm25 uses Counter(query), so repetition increases contribution.
        boosted = []
        for tok in query_tokens:
            boosted.append(tok)
            if _HAS_DIGIT.search(tok):
                boosted.extend([tok] * (_NUMERIC_BOOST - 1))
        scores = self.index.get_scores(boosted)

        eligible = range(len(self.obosn_list))
        if entity_filter is not None:
            allowed = set(entity_filter)
            eligible = [i for i in eligible if self.entity_types[i] in allowed]
        if group_filter is not None:
            allowed_g = set(group_filter)
            eligible = [
                i for i in eligible if self.rate_group_ids[i] in allowed_g
            ]

        eligible = sorted(eligible, key=lambda i: scores[i], reverse=True)[:top_k]
        out: List[Dict] = []
        rank = 0
        for i in eligible:
            if scores[i] <= 0:
                continue
            rank += 1
            out.append({
                "obosn": self.obosn_list[i],
                "naim": self.naim_list[i],
                "score": float(scores[i]),
                "rank": rank,
                "entity_type": self.entity_types[i],
                "rate_group_id": self.rate_group_ids[i],
            })
        return out
