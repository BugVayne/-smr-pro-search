"""Ingest historical XML estimates into the SQLite store.

Drop any number of XML files (and optionally subfolders) into
``data/raw/`` and run::

    python -m training.ingest_xml          # uses data/raw/ recursively
    python -m training.ingest_xml path/    # explicit path

Deduplication strategy
----------------------
Deduplication is **file-level**, not content-level:

* The same file is never imported twice (tracked by SHA-256 of its
  content).
* Two *different* files that happen to contain a PTM with identical
  codes are stored as **two separate transactions** — this is correct,
  because each file represents an independent project.  Collapsing them
  would undercount the real frequency and break FP-Growth support values.

Re-running the script is safe: already-processed files are skipped and
the summary shows how many were new vs. skipped.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict

from common.config import RAW_DIR
from common.db import (
    insert_transaction,
    is_file_ingested,
    mark_file_ingested,
    upsert_rascenki,
)
from common.logging_config import get_logger
from training.xml_parser import discover_xml_files, file_hash, parse_xml

log = get_logger(__name__)


TIP_TO_ENTITY = {
    "100": "rate",
    "101": "material",
    "103": "mechanism",
}


def ingest(xml_dir: Path, recursive: bool = True) -> Dict[str, int]:
    """Ingest every new XML file under ``xml_dir``.

    Returns a stats dict::

        {
            "files_total":   N,
            "files_new":     N,   # processed this run
            "files_skipped": N,   # already in DB
            "ptms":          N,
            "rascenki":      N,
            "transactions":  N,
        }
    """
    files = discover_xml_files(xml_dir, recursive=recursive)
    if not files:
        log.warning("No XML files found under %s", xml_dir)
        return {k: 0 for k in
                ("files_total", "files_new", "files_skipped",
                 "ptms", "rascenki", "transactions")}

    rascenki_seen: Dict[str, dict] = {}
    files_new = files_skipped = ptms_total = tx_total = 0

    for idx, path in enumerate(files, 1):
        fhash = file_hash(path)

        if is_file_ingested(fhash):
            log.info("[%d/%d] SKIP (already imported) %s",
                     idx, len(files),
                     path.relative_to(xml_dir) if path.is_relative_to(xml_dir) else path)
            files_skipped += 1
            continue

        rel = path.relative_to(xml_dir) if path.is_relative_to(xml_dir) else path
        log.info("[%d/%d] %s", idx, len(files), rel)

        try:
            ptms_in_file = 0
            for ptm in parse_xml(path):
                # Collect rascenki for the catalogue
                for r in ptm.rascenki:
                    if r.obosn not in rascenki_seen:
                        rascenki_seen[r.obosn] = {
                            "obosn": r.obosn,
                            "naim": r.naim,
                            "tip": r.tip,
                            "ed_izm": r.ed_izm or "",
                            "entity_type": TIP_TO_ENTITY.get(r.tip, "rate"),
                        }

                # Every PTM → one transaction (no content dedup)
                items = [r.obosn for r in ptm.rascenki]
                if items:
                    insert_transaction(
                        ptm.ptm_kod, ptm.ptm_naim, ptm.glava,
                        items, file_hash=fhash,
                    )
                    ptms_in_file += 1

            mark_file_ingested(fhash, str(path), ptms_in_file)
            log.info("        → %d PTMs", ptms_in_file)
            files_new += 1
            ptms_total += ptms_in_file
            tx_total += ptms_in_file

        except Exception as exc:
            log.warning("        ✗ failed: %s", exc)

    # Bulk upsert catalogue at the end
    if rascenki_seen:
        upsert_rascenki(rascenki_seen.values())

    stats = {
        "files_total":   len(files),
        "files_new":     files_new,
        "files_skipped": files_skipped,
        "ptms":          ptms_total,
        "rascenki":      len(rascenki_seen),
        "transactions":  tx_total,
    }
    log.info("=" * 60)
    log.info("Ingestion summary")
    log.info("=" * 60)
    log.info("  Files found         : %d", stats["files_total"])
    log.info("  Files imported      : %d", stats["files_new"])
    log.info("  Files skipped       : %d (already in DB)", stats["files_skipped"])
    log.info("  PTMs / transactions : %d", stats["ptms"])
    log.info("  Unique rascenki     : %d", stats["rascenki"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest XML estimates from a directory (recursively)."
    )
    parser.add_argument(
        "xml_dir", nargs="?", default=str(RAW_DIR),
        help=f"Directory to scan (default: {RAW_DIR}). Searched recursively.",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Disable recursive scan; only the top level is read.",
    )
    args = parser.parse_args()
    ingest(Path(args.xml_dir), recursive=not args.no_recursive)


if __name__ == "__main__":
    main()
