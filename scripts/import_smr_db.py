"""Import the SMR-Pro catalogue from an external SQLite database.

Reads four tables of the source DB and writes them into our search-side
``rascenka`` and ``rate_groups`` tables. The source DB is treated as
read-only and never modified.

Mapping
-------
    Source table       →  rascenka.entity_type
    ------------------    ---------------------
    Rates                 'rate'
    Materials             'material'
    Mechanisms            'mechanism'
    Devices               'device'

For ``Rates`` we also remember ``RateGroupId`` so that searches can
narrow the result set to a specific group (or augment query text with
the group name for better recall).

Usage
-----

    python -m scripts.import_smr_db                          # default path
    python -m scripts.import_smr_db data/raw/smr_catalog.db  # explicit
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Iterable, Iterator

from common.config import RAW_DIR
from common.db import init_schema, upsert_rascenki, upsert_rate_groups
from common.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_SOURCE = RAW_DIR / "smr_catalog.db"
CHUNK = 1000


def _connect_source(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(
            f"Source DB not found: {path}\n"
            f"Place your SMR-Pro SQLite file at this location, or pass a "
            f"different path as the first argument."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _chunked(it: Iterable, n: int = CHUNK):
    buf = []
    for row in it:
        buf.append(row)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

def import_rate_groups(src: sqlite3.Connection) -> int:
    cur = src.execute(
        "SELECT Id, Code, Name, NamePrefix, ParentId FROM RateGroups"
    )
    rows = [
        {
            "id": r["Id"],
            "code": r["Code"],
            "name": r["Name"],
            "name_prefix": r["NamePrefix"],
            "parent_id": r["ParentId"],
        }
        for r in cur
    ]
    upsert_rate_groups(rows)
    log.info("Imported %d rate groups", len(rows))
    return len(rows)


def _iter_rates(src: sqlite3.Connection) -> Iterator[dict]:
    cur = src.execute(
        "SELECT Id, Code, Name, UnitsOfMeasurement, RateGroupId FROM Rates"
    )
    skipped_star = 0
    all_codes: set[str] = set()

    rows = cur.fetchall()
    for r in rows:
        all_codes.add(r["Code"])

    for r in rows:
        code: str = r["Code"]
        # Skip starred duplicates (e.g. "А1-15-1*") when the base code exists.
        # The * suffix means "with complicating conditions" — same work item,
        # not a distinct rascenka. Keep it only if the base is absent.
        if code.endswith("*") and code[:-1] in all_codes:
            skipped_star += 1
            continue
        yield {
            "obosn": code,
            "naim": r["Name"],
            "tip": "100",
            "ed_izm": r["UnitsOfMeasurement"],
            "entity_type": "rate",
            "rate_group_id": r["RateGroupId"],
            "source_id": r["Id"],
        }

    if skipped_star:
        log.info("Skipped %d starred duplicate rates (* suffix with existing base)",
                 skipped_star)


def _iter_materials(src: sqlite3.Connection) -> Iterator[dict]:
    cur = src.execute(
        "SELECT Id, Code, FormattedCode, Name, UnitsOfMeasurement, entity_type "
        "FROM Materials"
    )
    for r in cur:
        obosn = r["Code"] or r["FormattedCode"]
        yield {
            "obosn": obosn,
            "naim": r["Name"],
            "tip": "101",
            "ed_izm": r["UnitsOfMeasurement"],
            "entity_type": "material",
            "source_id": r["Id"],
        }


def _iter_mechanisms(src: sqlite3.Connection) -> Iterator[dict]:
    cur = src.execute(
        "SELECT Id, Code, Name, UnitsOfMeasurement FROM Mechanisms"
    )
    for r in cur:
        yield {
            "obosn": r["Code"],
            "naim": r["Name"],
            "tip": "103",
            "ed_izm": r["UnitsOfMeasurement"],
            "entity_type": "mechanism",
            "source_id": r["Id"],
        }


def _iter_devices(src: sqlite3.Connection) -> Iterator[dict]:
    cur = src.execute("SELECT Id, Code, Name FROM Devices")
    for r in cur:
        yield {
            "obosn": r["Code"],
            "naim": r["Name"],
            "tip": "105",
            "ed_izm": None,
            "entity_type": "device",
            "source_id": r["Id"],
        }


def import_entity(name: str, iterator: Iterator[dict]) -> int:
    """Chunk-stream an entity type into our rascenka table."""
    total = 0
    for chunk in _chunked(iterator):
        upsert_rascenki(chunk)
        total += len(chunk)
    log.info("Imported %d %s", total, name)
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def import_all(source_path: Path) -> dict:
    init_schema()
    log.info("Importing from %s", source_path)

    with closing(_connect_source(source_path)) as src:
        stats = {
            "rate_groups": import_rate_groups(src),
            "rates": import_entity("rates", _iter_rates(src)),
            "materials": import_entity("materials", _iter_materials(src)),
            "mechanisms": import_entity("mechanisms", _iter_mechanisms(src)),
            "devices": import_entity("devices", _iter_devices(src)),
        }
    log.info("Import finished: %s", stats)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import the SMR-Pro catalogue from a SQLite DB."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_SOURCE),
        help=f"Path to the source SQLite DB (default: {DEFAULT_SOURCE})",
    )
    args = parser.parse_args()
    try:
        import_all(Path(args.source))
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
