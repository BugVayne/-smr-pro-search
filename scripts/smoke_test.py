"""End-to-end smoke test.

Probes the health endpoint of every microservice, then sends a couple
of search queries through the gateway and verifies a sensible response
is returned.
"""
from __future__ import annotations

import sys
import time
from typing import Tuple

import requests

from common.config import (
    GATEWAY_URL,
    POSTPROCESS_URL,
    PREPROCESS_URL,
    RANKING_URL,
    RETRIEVAL_URL,
)
from common.logging_config import get_logger

log = get_logger("smoke")

SERVICES: Tuple[Tuple[str, str], ...] = (
    ("preprocessing",  PREPROCESS_URL),
    ("retrieval",      RETRIEVAL_URL),
    ("ranking",        RANKING_URL),
    ("postprocessing", POSTPROCESS_URL),
    ("gateway",        GATEWAY_URL),
)

QUERIES = [
    "тёплый пол",
    "покраска стен",
    "кровельный пирог",
    "штукатурка",
    "ГЭСН07-05-011",
]


def check_health() -> bool:
    ok = True
    for name, url in SERVICES:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            r.raise_for_status()
            log.info("  ✓ %-15s %s", name, r.json())
        except Exception as exc:
            log.error("  ✗ %-15s %s", name, exc)
            ok = False
    return ok


def run_search(query: str) -> bool:
    t = time.perf_counter()
    try:
        r = requests.post(
            f"{GATEWAY_URL}/search",
            json={"query": query, "context": {}},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.error("  ✗ %r: %s", query, exc)
        return False

    elapsed = (time.perf_counter() - t) * 1000
    items = data.get("items", [])
    kits = data.get("kits", [])
    log.info(
        "  → %r: %d items, %d kits, %.0fms (corrected=%r)",
        query, len(items), len(kits), elapsed, data.get("corrected"),
    )
    if items:
        top = items[0]
        log.info("    top: %s %s (score=%.3f)",
                 top["obosn"], top["naim"][:60], top["score"])
    return True


def main() -> int:
    log.info("=== Health checks ===")
    if not check_health():
        log.error("Some services are down – aborting.")
        return 2

    log.info("=== Search queries ===")
    failed = 0
    for q in QUERIES:
        if not run_search(q):
            failed += 1

    log.info("Done. %d/%d queries succeeded.", len(QUERIES) - failed, len(QUERIES))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
