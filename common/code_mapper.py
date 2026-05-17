"""2012 → 2022 material code translator.

Loads ``materialCodesDictionary.csv`` once and exposes a fast lookup.
The CSV has three columns:
    OBOSN  – old 2012 base code
    NAIM   – human-readable name (not used here)
    KOD22  – new 2022 base code (may be empty)

Rules:
* If a code is in the dict and KOD22 is non-empty → return KOD22.
* Otherwise → return the original code unchanged.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from common.config import CODE_DICT_PATH
from common.logging_config import get_logger

log = get_logger(__name__)


class CodeMapper:
    def __init__(self, csv_path: Optional[Path] = None):
        self._map: Dict[str, str] = {}
        path = Path(csv_path) if csv_path else CODE_DICT_PATH
        if not path.exists():
            log.warning("Code dictionary not found at %s – code translation disabled", path)
            return
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                for row in csv.DictReader(f):
                    old = row["OBOSN"].strip()
                    new = row["KOD22"].strip()
                    if old and new:
                        self._map[old] = new
            log.info("Loaded code dictionary: %d 2012->2022 mappings from %s",
                     len(self._map), path)
        except Exception as exc:
            log.warning("Could not load code dictionary: %s", exc)

    def translate(self, code: str) -> str:
        """Return 2022 code if mapping exists, otherwise return original."""
        return self._map.get(code, code)

    def translate_list(self, codes: list[str]) -> tuple[list[str], int]:
        """Translate a list of codes. Returns (translated_list, n_translated)."""
        result = []
        n = 0
        for c in codes:
            t = self._map.get(c)
            if t:
                result.append(t)
                n += 1
            else:
                result.append(c)
        return result, n

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, code: str) -> bool:
        return code in self._map


@lru_cache(maxsize=1)
def get_mapper() -> CodeMapper:
    """Return a process-level singleton mapper."""
    return CodeMapper()
