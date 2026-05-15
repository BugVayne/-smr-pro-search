"""Initialise the SQLite schema."""
from __future__ import annotations

from common.db import init_schema
from common.logging_config import get_logger

log = get_logger(__name__)


if __name__ == "__main__":
    init_schema()
    log.info("Schema initialised.")
