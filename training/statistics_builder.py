"""Compute the statistical profile for each rascenka.

Profile = {
    "f_abs":  absolute count of PTMs containing the rascenka,
    "f_norm": f_abs / total_PTMs,
    "ctx":    {ptm_kod: count},
    "glava":  {glava_name: count},
}
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from common.db import all_transactions, replace_profiles
from common.logging_config import get_logger

log = get_logger(__name__)


def build() -> int:
    transactions = all_transactions()
    if not transactions:
        log.warning("No transactions – skipping profile build")
        replace_profiles([])
        return 0

    total = len(transactions)
    f_abs: Counter[str] = Counter()
    ctx_count: Dict[str, Counter[str]] = defaultdict(Counter)
    glava_count: Dict[str, Counter[str]] = defaultdict(Counter)

    for tx in transactions:
        kod = tx["ptm_kod"] or ""
        gl = tx["glava"] or ""
        for obosn in tx["items"]:
            f_abs[obosn] += 1
            ctx_count[obosn][kod] += 1
            glava_count[obosn][gl] += 1

    profiles: List[dict] = []
    for obosn, count in f_abs.items():
        profiles.append({
            "obosn": obosn,
            "f_abs": int(count),
            "f_norm": float(count) / total,
            "ctx": dict(ctx_count[obosn]),
            "glava": dict(glava_count[obosn]),
        })

    replace_profiles(profiles)
    log.info("Stored %d rascenka profiles (over %d PTMs)", len(profiles), total)
    return len(profiles)


if __name__ == "__main__":
    build()
