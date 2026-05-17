"""Build the PTM retrieval index.

The index answers: "given query tokens, which rascenki are typically
used in PTMs whose name matches those tokens?"

Sources:
    1. All transactions in the DB  (historical, 54k rows)
    2. ptms.csv                    (canonical names, 596 rows)

When a transaction's ptm_kod is in ptms.csv, the canonical name from
the CSV is used instead of (or alongside) the user-written ptm_naim,
so both predefined and custom PTMs are searchable.

Index structure (saved as pickle):
    {
        "ptm_list": [
            {
                "kod":       str,
                "naim":      str,
                "tokens":    List[str],   # lemmatized name
                "rascenki":  [            # sorted by freq desc
                    {"obosn": str, "naim": str, "entity_type": str,
                     "rate_group_id": int|None, "freq": float},
                    ...
                ],
                "tx_count": int,          # number of transactions
            },
            ...
        ],
        "token_to_ptms": {token: [idx, ...], ...},
    }
"""
from __future__ import annotations

import csv
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from common.config import PTM_INDEX_PATH, RAW_DIR
from common.db import all_rascenki, all_transactions
from common.logging_config import get_logger
from services.preprocessing.lemmatizer import lemmatize

log = get_logger(__name__)

_PTM_CSV = RAW_DIR / "ptms.csv"
_TOP_RASCENKI_PER_PTM = 50   # max rascenki stored per PTM entry
_MIN_TX_COUNT = 1            # PTMs with fewer transactions are still kept


# ---------------------------------------------------------------------------
# Load canonical PTM names from CSV
# ---------------------------------------------------------------------------

def _load_csv_ptms() -> Dict[str, str]:
    """Return {kod: canonical_naim} from ptms.csv."""
    if not _PTM_CSV.exists():
        log.warning("ptms.csv not found at %s – canonical names unavailable", _PTM_CSV)
        return {}
    result: Dict[str, str] = {}
    try:
        with open(_PTM_CSV, encoding="utf-8-sig", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                parts = row[1].split("#")
                kod  = parts[0].strip()
                naim = parts[2].strip() if len(parts) > 2 else ""
                if kod and naim:
                    result[kod] = naim
    except Exception as exc:
        log.warning("Could not read ptms.csv: %s", exc)
    log.info("Loaded %d canonical PTM names from ptms.csv", len(result))
    return result


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_ptm_index() -> int:
    csv_ptms = _load_csv_ptms()

    transactions = all_transactions()
    if not transactions:
        log.warning("No transactions – PTM index not built")
        return 0

    # Build rascenka metadata lookup {obosn: {naim, entity_type, rate_group_id}}
    rascenka_meta: Dict[str, dict] = {
        r["obosn"]: r for r in all_rascenki()
    }

    # Group transactions by a (kod, naim) key.
    # Prefer the canonical CSV name when available.
    ptm_items: Dict[tuple, Counter] = defaultdict(Counter)
    ptm_naim_by_key: Dict[tuple, str] = {}

    for tx in transactions:
        kod  = (tx["ptm_kod"] or "").strip()
        naim = (tx["ptm_naim"] or "").strip()

        # Use canonical name from CSV when the code is known
        canonical = csv_ptms.get(kod, "")
        effective_naim = canonical or naim

        if not effective_naim:
            continue

        key = (kod, effective_naim)
        ptm_naim_by_key[key] = effective_naim
        for obosn in tx["items"]:
            ptm_items[key][obosn] += 1

    # Also add CSV entries that have NO transactions yet
    # (so they're at least searchable even if they return 0 rascenki)
    for kod, naim in csv_ptms.items():
        key = (kod, naim)
        if key not in ptm_items:
            ptm_items[key] = Counter()
            ptm_naim_by_key[key] = naim

    log.info("Unique PTM entries (kod+naim): %d", len(ptm_items))

    # Build the list of PTM records
    ptm_list: List[dict] = []
    token_to_ptms: Dict[str, List[int]] = defaultdict(list)

    for (kod, naim), rascenki_counts in ptm_items.items():
        tokens = lemmatize(naim)
        if not tokens:
            continue

        tx_count = sum(rascenki_counts.values())
        total_items = tx_count or 1  # avoid div-by-zero

        # Top rascenki by frequency within this PTM
        sorted_r = rascenki_counts.most_common(_TOP_RASCENKI_PER_PTM)
        rascenki_records = []
        for obosn, count in sorted_r:
            meta = rascenka_meta.get(obosn, {})
            rascenki_records.append({
                "obosn":         obosn,
                "naim":          meta.get("naim", obosn),
                "entity_type":   meta.get("entity_type", "rate"),
                "rate_group_id": meta.get("rate_group_id"),
                "freq":          count / total_items,
            })

        idx = len(ptm_list)
        ptm_list.append({
            "kod":       kod,
            "naim":      naim,
            "tokens":    tokens,
            "rascenki":  rascenki_records,
            "tx_count":  len([tx for tx in transactions
                               if (tx.get("ptm_kod") or "").strip() == kod
                               or (tx.get("ptm_naim") or "").strip() == naim]),
        })

        for tok in set(tokens):
            token_to_ptms[tok].append(idx)

    PTM_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PTM_INDEX_PATH, "wb") as f:
        pickle.dump({
            "ptm_list":      ptm_list,
            "token_to_ptms": dict(token_to_ptms),
        }, f)

    log.info("Built PTM index: %d entries, %d tokens -> %s",
             len(ptm_list), len(token_to_ptms), PTM_INDEX_PATH)
    return len(ptm_list)


if __name__ == "__main__":
    build_ptm_index()
