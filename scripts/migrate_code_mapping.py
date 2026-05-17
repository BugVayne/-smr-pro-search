"""Apply 2012 → 2022 code translation to existing transactions in the DB.

Run once after adding materialCodesDictionary.csv:

    python scripts/migrate_code_mapping.py
    python scripts/migrate_code_mapping.py --dry-run   # preview only
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from common.code_mapper import CodeMapper
from common.config import CODE_DICT_PATH
from common.db import db
from common.logging_config import get_logger

log = get_logger("migrate_code_mapping")


def migrate(dry_run: bool = False) -> None:
    mapper = CodeMapper(CODE_DICT_PATH)
    if len(mapper) == 0:
        log.error("Code dictionary is empty – aborting")
        return

    log.info("Loaded %d code mappings", len(mapper))
    log.info("dry_run=%s", dry_run)

    with db() as conn:
        rows = conn.execute("SELECT id, items FROM transactions").fetchall()
    log.info("Transactions to check: %d", len(rows))

    updated_rows = []
    total_translated = 0
    freq: Counter = Counter()

    for row in rows:
        tx_id = row["id"]
        items = json.loads(row["items"])
        new_items, n = mapper.translate_list(items)
        if n > 0:
            for old, new in zip(items, new_items):
                if old != new:
                    freq[old] += 1
            total_translated += n
            updated_rows.append((json.dumps(new_items, ensure_ascii=False), tx_id))

    log.info("Transactions needing update : %d / %d", len(updated_rows), len(rows))
    log.info("Individual code translations: %d", total_translated)
    log.info("Most-translated old codes   : %s",
             freq.most_common(10))

    if dry_run:
        log.info("DRY RUN - no changes written")
        return

    if not updated_rows:
        log.info("Nothing to update")
        return

    with db() as conn:
        conn.executemany(
            "UPDATE transactions SET items = ? WHERE id = ?",
            updated_rows,
        )
    log.info("Done – %d transactions updated", len(updated_rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing to DB")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
