"""Train a FastText model on the corpus of rascenka names.

This is used in the preprocessing layer for query expansion. FastText is
preferred over Word2Vec because of subword robustness to OOV and typos.

Corpus sources (in priority order):
  1. Rascenka names from the 2022 catalogue (main signal)
  2. PTM names from transactions (ptm_naim field)
  3. Canonical PTM names + descriptions from ptms.csv (new: adds
     conceptual-level vocabulary missing from rascenka names alone)
"""
from __future__ import annotations

import csv
from typing import List

from common.config import FASTTEXT_MODEL_PATH, RAW_DIR
from common.db import all_rascenki, all_transactions
from common.logging_config import get_logger
from services.preprocessing.lemmatizer import tokenize

log = get_logger(__name__)

PTM_CSV_PATH = RAW_DIR / "ptms.csv"


def _load_ptm_csv_sentences() -> List[List[str]]:
    """Load names and descriptions from the predefined PTM catalogue CSV."""
    if not PTM_CSV_PATH.exists():
        return []
    sentences = []
    try:
        with open(PTM_CSV_PATH, encoding="utf-8-sig", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                parts = row[1].split("#")
                naim = parts[2].strip() if len(parts) > 2 else ""
                desc = parts[3].strip() if len(parts) > 3 else ""
                if naim:
                    sentences.append(tokenize(naim))
                if desc:
                    sentences.append(tokenize(desc))
    except Exception as exc:
        log.warning("Could not read PTM CSV: %s", exc)
    return sentences


def _corpus() -> List[List[str]]:
    sentences: List[List[str]] = []
    for r in all_rascenki():
        sentences.append(tokenize(r["naim"]))
    tx_naims_seen: set = set()
    for tx in all_transactions():
        naim = tx["ptm_naim"]
        if naim and naim not in tx_naims_seen:
            tx_naims_seen.add(naim)
            sentences.append(tokenize(naim))
    ptm_csv_sentences = _load_ptm_csv_sentences()
    sentences.extend(ptm_csv_sentences)
    log.info("Corpus: %d rascenki + %d unique tx PTMs + %d CSV PTM sentences",
             len(list(all_rascenki())), len(tx_naims_seen), len(ptm_csv_sentences))
    return [s for s in sentences if s]


def train(vector_size: int = 100, window: int = 5, min_count: int = 1, epochs: int = 10) -> bool:
    try:
        from gensim.models import FastText
    except ImportError:
        log.warning("gensim not installed – skipping FastText")
        return False

    corpus = _corpus()
    if len(corpus) < 5:
        log.warning("Corpus too small (%d) – skipping FastText", len(corpus))
        return False

    log.info("Training FastText on %d sentences", len(corpus))
    model = FastText(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=1,
    )
    FASTTEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(FASTTEXT_MODEL_PATH))
    log.info("Saved FastText to %s", FASTTEXT_MODEL_PATH)
    return True


if __name__ == "__main__":
    train()
