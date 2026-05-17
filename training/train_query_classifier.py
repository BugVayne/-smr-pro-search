"""Train a query-type classifier (atomic vs aggregated).

Training data is built automatically from existing project data:

    AGGREGATED examples:
        - canonical PTM names from ptms.csv          (~596)
        - clean user-written PTM names from transactions (~15k)

    ATOMIC examples:
        - rascenka names, first 4 words (simulate user query)  (~20k sample)

The model is a sklearn Pipeline:
    TF-IDF (char wb n-grams 2-4)  ->  LogisticRegression

Saved to data/models/query_type_classifier.pkl so the existing
QueryClassifier in services/preprocessing/classifier.py loads it
automatically on next service start.
"""
from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path
from typing import List, Tuple

from common.config import QTYPE_MODEL_PATH, RAW_DIR
from common.db import all_rascenki, all_transactions
from common.logging_config import get_logger

log = get_logger(__name__)

PTM_CSV_PATH = RAW_DIR / "ptms.csv"

# Tuning
_MAX_RASCENKA_WORDS = 4   # take first N words from rascenka name to simulate atomic query
_ATOMIC_SAMPLE = 20_000   # cap atomic examples so training stays fast
_TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _load_csv_ptm_names() -> List[str]:
    if not PTM_CSV_PATH.exists():
        return []
    names = []
    with open(PTM_CSV_PATH, encoding="utf-8-sig", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            parts = row[1].split("#")
            naim = parts[2].strip() if len(parts) > 2 else ""
            if naim:
                names.append(naim)
    return names


def _clean_tx_name(naim: str) -> bool:
    """Return True if the name looks like a real PTM concept (not a code or junk)."""
    if not naim:
        return False
    words = naim.split()
    if len(words) < 2 or len(words) > 12:
        return False
    if re.search(r"\d{4,}", naim):   # long digit runs = project-specific
        return False
    if len(naim) > 200:
        return False
    return True


def _subphrases(name: str, min_w: int = 2, max_w: int = 4) -> List[str]:
    """Extract all contiguous word subphrases from a name."""
    words = name.split()
    out = []
    for length in range(min_w, min(max_w + 1, len(words) + 1)):
        for start in range(len(words) - length + 1):
            out.append(" ".join(words[start:start + length]))
    return out


def _collect_aggregated() -> List[str]:
    csv_names = _load_csv_ptm_names()
    seen = set(n.lower() for n in csv_names)
    aggregated = list(csv_names)

    # Add 2-4 word subphrases from CSV names so the model sees
    # short concept-like phrases (e.g. "тёплый пол" from longer names)
    for name in csv_names:
        for sub in _subphrases(name):
            if sub.lower() not in seen:
                seen.add(sub.lower())
                aggregated.append(sub)

    n_csv_with_subs = len(aggregated)

    for tx in all_transactions():
        naim = (tx["ptm_naim"] or "").strip()
        if _clean_tx_name(naim) and naim.lower() not in seen:
            seen.add(naim.lower())
            aggregated.append(naim)

    log.info("Aggregated examples: %d (%d CSV+subphrases + %d from transactions)",
             len(aggregated), n_csv_with_subs, len(aggregated) - n_csv_with_subs)
    return aggregated


def _collect_atomic() -> List[str]:
    seen: set = set()
    atomic: List[str] = []
    for r in all_rascenki():
        words = r["naim"].split()[:_MAX_RASCENKA_WORDS]
        phrase = " ".join(words).strip()
        if len(phrase.split()) >= 2 and phrase.lower() not in seen:
            seen.add(phrase.lower())
            atomic.append(phrase)
        if len(atomic) >= _ATOMIC_SAMPLE:
            break
    log.info("Atomic examples: %d (capped at %d)", len(atomic), _ATOMIC_SAMPLE)
    return atomic


def build_dataset() -> Tuple[List[str], List[str]]:
    aggregated = _collect_aggregated()
    atomic = _collect_atomic()
    texts  = aggregated + atomic
    labels = ["aggregated"] * len(aggregated) + ["atomic"] * len(atomic)
    return texts, labels


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train() -> bool:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError:
        log.warning("scikit-learn not installed – skipping query classifier training")
        return False

    texts, labels = build_dataset()
    if len(set(labels)) < 2:
        log.warning("Not enough class diversity – skipping")
        return False

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=_TEST_SIZE,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=50_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )),
    ])

    log.info("Training on %d examples (%d train / %d test)",
             len(texts), len(X_train), len(X_test))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    log.info("Classification report:\n%s", report)

    QTYPE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QTYPE_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info("Saved query-type classifier to %s", QTYPE_MODEL_PATH)
    return True


if __name__ == "__main__":
    train()
