"""End-to-end training pipeline.

Order matters:

    0. import catalogue from external SMR-Pro SQLite (if present)
    1. ingest XML estimates into transactions
    2. build statistical profiles
    3. run FP-Growth
    4. train FastText
    5. fine-tune SBERT on (ptm, rascenka) pairs
    6. build BM25 + FAISS indices (FAISS needs the new SBERT)
    7. train CatBoost ranker if we have user feedback

Usage:

    python -m training.pipeline
    python -m training.pipeline --xml-dir data/raw
    python -m training.pipeline --skip-sbert     # quick dev cycle
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common.config import RAW_DIR
from common.db import init_schema, rascenki_count_by_type, transactions_count
from common.logging_config import get_logger
from scripts import import_smr_db
from training import (
    build_indices,
    ingest_xml,
    statistics_builder,
    train_catboost,
    train_fasttext,
    train_fp_growth,
    train_query_classifier,
    train_sbert,
)

log = get_logger("training.pipeline")


def run(
    xml_dir: Path = RAW_DIR,
    catalog_db: Path | None = None,
    skip_sbert: bool = False,
    skip_fasttext: bool = False,
    skip_catboost: bool = False,
    skip_faiss: bool = False,
) -> None:
    log.info("=" * 60)
    log.info("Adaptive knowledge-base training pipeline")
    log.info("=" * 60)

    init_schema()

    # 0. Catalog import (if a source DB is provided)
    if catalog_db is None:
        catalog_db = RAW_DIR / "smr_catalog.db"
    if Path(catalog_db).exists():
        log.info("[0/7] Importing catalogue from %s", catalog_db)
        import_smr_db.import_all(Path(catalog_db))
    else:
        log.info("[0/7] No catalogue DB at %s – skipping import", catalog_db)

    log.info("Catalogue counts: %s", rascenki_count_by_type())

    log.info("[1/7] Ingesting XML estimates from %s", xml_dir)
    if Path(xml_dir).exists():
        ingest_xml.ingest(Path(xml_dir))
    else:
        log.warning("XML dir %s does not exist – skipping ingestion", xml_dir)

    log.info("Transactions in store: %d", transactions_count())

    log.info("[2/7] Building statistical profiles")
    statistics_builder.build()

    log.info("[3/7] Running FP-Growth")
    train_fp_growth.train()

    if not skip_fasttext:
        log.info("[4/7] Training FastText")
        train_fasttext.train()
    else:
        log.info("[4/7] Skipping FastText")

    if not skip_sbert:
        log.info("[5/7] Fine-tuning SBERT")
        train_sbert.train()
    else:
        log.info("[5/7] Skipping SBERT fine-tune")

    log.info("[6/8] Building BM25 + FAISS + SymSpell + PTM index")
    build_indices.build_all(skip_faiss=skip_faiss or skip_sbert)

    log.info("[7/8] Training query-type classifier")
    train_query_classifier.train()

    if not skip_catboost:
        log.info("[8/8] Training CatBoost ranker")
        train_catboost.train()
    else:
        log.info("[8/8] Skipping CatBoost")

    log.info("Training pipeline finished.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", default=str(RAW_DIR))
    parser.add_argument("--catalog-db", default=None,
                        help="Path to the source SMR-Pro SQLite file. "
                             "Default: data/raw/smr_catalog.db")
    parser.add_argument("--skip-sbert", action="store_true")
    parser.add_argument("--skip-fasttext", action="store_true")
    parser.add_argument("--skip-catboost", action="store_true")
    parser.add_argument("--skip-faiss", action="store_true")
    args = parser.parse_args()

    run(
        xml_dir=Path(args.xml_dir),
        catalog_db=Path(args.catalog_db) if args.catalog_db else None,
        skip_sbert=args.skip_sbert,
        skip_fasttext=args.skip_fasttext,
        skip_catboost=args.skip_catboost,
        skip_faiss=args.skip_faiss,
    )


if __name__ == "__main__":
    main()
