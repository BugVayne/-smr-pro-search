"""FP-Growth training over the transactional base.

Reads PTM transactions from SQLite and uses ``mlxtend`` to build:

1. frequent itemsets (with support >= FP_SUPPORT_MIN)
2. association rules (confidence >= FP_CONFIDENCE_MIN, lift >= 1)

Rules are written back to the ``association_rules`` table.
"""
from __future__ import annotations

from typing import List

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from common.config import (
    FP_CONFIDENCE_MIN,
    FP_LIFT_MIN,
    FP_MAX_LEN,
    FP_SUPPORT_MIN,
)
from common.db import all_transactions, replace_rules
from common.logging_config import get_logger

log = get_logger(__name__)


def train() -> int:
    """Run FP-Growth and persist rules. Returns number of rules stored."""
    transactions = [tx["items"] for tx in all_transactions()]
    if len(transactions) < 5:
        log.warning("Only %d transactions – FP-Growth will yield few rules", len(transactions))
        if not transactions:
            replace_rules([])
            return 0

    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    log.info("Running FP-Growth: %d txs, %d items, min_support=%.3f",
             len(transactions), df.shape[1], FP_SUPPORT_MIN)

    frequent = fpgrowth(
        df,
        min_support=FP_SUPPORT_MIN,
        use_colnames=True,
        max_len=FP_MAX_LEN,
    )
    if frequent.empty:
        log.warning("No frequent itemsets found")
        replace_rules([])
        return 0

    rules_df = association_rules(
        frequent, metric="confidence", min_threshold=FP_CONFIDENCE_MIN
    )
    rules_df = rules_df[rules_df["lift"] >= FP_LIFT_MIN]

    rules: List[dict] = []
    for _, row in rules_df.iterrows():
        rules.append({
            "antecedent": list(row["antecedents"]),
            "consequent": list(row["consequents"]),
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
        })

    replace_rules(rules)
    log.info("Stored %d association rules", len(rules))
    return len(rules)


if __name__ == "__main__":
    train()
