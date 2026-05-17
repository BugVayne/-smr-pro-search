"""PTM-based retrieval.

Answers: "which rascenki are typically used in PTMs whose name
matches the query?"

Algorithm:
  1. Tokenize query lemmas
  2. Look each token up in the inverted index -> matching PTM indices
  3. Score each PTM by overlap fraction: matched_tokens / ptm_tokens
  4. Take top-5 PTMs, aggregate their rascenki weighted by ptm_score * freq
  5. Return ranked rascenki list

The result dict format matches BM25Search.search() output so it can be
passed directly into rrf_fusion.rrf().
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from common.config import PTM_INDEX_PATH
from common.logging_config import get_logger

log = get_logger(__name__)

_TOP_PTMS = 5        # how many matching PTMs to aggregate
_MIN_OVERLAP = 0.4   # minimum token overlap fraction to consider a PTM match


class PTMSearch:
    def __init__(self, index_path: Optional[Path] = None):
        self.ptm_list: List[dict] = []
        self.token_to_ptms: Dict[str, List[int]] = {}
        path = index_path or PTM_INDEX_PATH

        if not Path(path).exists():
            log.warning("PTM index %s not found – run training", path)
            return

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            self.ptm_list = payload["ptm_list"]
            self.token_to_ptms = payload["token_to_ptms"]
            log.info("Loaded PTM index: %d entries, %d tokens",
                     len(self.ptm_list), len(self.token_to_ptms))
        except Exception as exc:
            log.warning("Could not load PTM index: %s", exc)

    def search(self, query_lemmas: List[str], top_k: int = 100) -> List[Dict]:
        """Return rascenki typical for PTMs matching query_lemmas.

        Return format (same as BM25Search):
            [{"obosn", "naim", "score", "rank", "entity_type", "rate_group_id"}, ...]
        """
        if not self.ptm_list or not query_lemmas:
            return []

        # --- Step 1: count token hits per PTM ---
        ptm_hits: Counter = Counter()
        for tok in query_lemmas:
            for idx in self.token_to_ptms.get(tok, []):
                ptm_hits[idx] += 1

        if not ptm_hits:
            return []

        # --- Step 2: score PTMs by overlap fraction ---
        ptm_scores: Dict[int, float] = {}
        for idx, hit_count in ptm_hits.items():
            n_tokens = max(len(self.ptm_list[idx]["tokens"]), 1)
            overlap = hit_count / n_tokens
            if overlap >= _MIN_OVERLAP:
                ptm_scores[idx] = overlap

        if not ptm_scores:
            return []

        # --- Step 3: aggregate rascenki from top PTMs ---
        rascenki_score: Dict[str, float] = {}
        rascenki_meta: Dict[str, dict] = {}

        top_ptms = sorted(ptm_scores.items(), key=lambda x: -x[1])[:_TOP_PTMS]
        for idx, ptm_score in top_ptms:
            ptm = self.ptm_list[idx]
            for r in ptm["rascenki"]:
                obosn = r["obosn"]
                rascenki_score[obosn] = (
                    rascenki_score.get(obosn, 0.0) + ptm_score * r["freq"]
                )
                if obosn not in rascenki_meta:
                    rascenki_meta[obosn] = r

        # --- Step 4: sort and return ---
        sorted_r = sorted(rascenki_score.items(), key=lambda x: -x[1])[:top_k]
        out = []
        for rank, (obosn, score) in enumerate(sorted_r, 1):
            meta = rascenki_meta[obosn]
            out.append({
                "obosn":         obosn,
                "naim":          meta["naim"],
                "score":         score,
                "rank":          rank,
                "entity_type":   meta.get("entity_type", "rate"),
                "rate_group_id": meta.get("rate_group_id"),
            })
        return out

    def matched_ptm_names(self, query_lemmas: List[str]) -> List[str]:
        """Return names of PTMs that matched (for logging)."""
        ptm_hits: Counter = Counter()
        for tok in query_lemmas:
            for idx in self.token_to_ptms.get(tok, []):
                ptm_hits[idx] += 1
        names = []
        for idx, hits in ptm_hits.most_common(_TOP_PTMS):
            n_tokens = max(len(self.ptm_list[idx]["tokens"]), 1)
            if hits / n_tokens >= _MIN_OVERLAP:
                names.append(self.ptm_list[idx]["naim"])
        return names

    def get_matched_ptms(
        self,
        query_lemmas: List[str],
        max_ptms: int = 3,
        max_members: int = 20,
    ) -> List[Dict]:
        """Return matched PTMs with their top расценки (for aggregated response).

        Return format:
            [{"ptm_naim": str, "overlap_score": float,
              "top_rascenki": [{"obosn", "naim", "entity_type", "rate_group_id"}, ...]}, ...]
        """
        if not self.ptm_list or not query_lemmas:
            return []

        ptm_hits: Counter = Counter()
        for tok in query_lemmas:
            for idx in self.token_to_ptms.get(tok, []):
                ptm_hits[idx] += 1

        results = []
        for idx, hit_count in ptm_hits.most_common(max_ptms * 3):
            n_tokens = max(len(self.ptm_list[idx]["tokens"]), 1)
            overlap = hit_count / n_tokens
            if overlap < _MIN_OVERLAP:
                continue
            ptm = self.ptm_list[idx]
            members = []
            for r in ptm["rascenki"][:max_members]:
                members.append({
                    "obosn":         r["obosn"],
                    "naim":          r["naim"],
                    "entity_type":   r.get("entity_type", "rate"),
                    "rate_group_id": r.get("rate_group_id"),
                })
            results.append({
                "ptm_naim":     ptm["naim"],
                "overlap_score": overlap,
                "top_rascenki": members,
            })
            if len(results) >= max_ptms:
                break

        return results
