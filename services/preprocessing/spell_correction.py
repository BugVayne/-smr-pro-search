"""Spell correction using SymSpell with a domain dictionary.

The dictionary is built from all unique tokens in the normative base
(see ``training.build_indices``). Falls back gracefully when SymSpell or
the dictionary is missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from common.config import SYMSPELL_DICT_PATH
from common.logging_config import get_logger

log = get_logger(__name__)

try:
    from symspellpy import SymSpell, Verbosity
except ImportError:  # symspell is optional
    SymSpell = None  # type: ignore
    Verbosity = None  # type: ignore


class SpellCorrector:
    def __init__(self, dict_path: Optional[Path] = None, max_edit_distance: int = 2):
        self.max_edit_distance = max_edit_distance
        self._sym = None

        if SymSpell is None:
            log.warning("symspellpy not installed – spell correction disabled")
            return

        dict_path = dict_path or SYMSPELL_DICT_PATH
        if not Path(dict_path).exists():
            log.warning(
                "Spell dictionary %s not found – run training to build it",
                dict_path,
            )
            return

        sym = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        sym.load_dictionary(str(dict_path), term_index=0, count_index=1, encoding="utf-8")
        self._sym = sym
        log.info("Loaded SymSpell dictionary from %s", dict_path)

    def correct_word(self, word: str) -> str:
        if not self._sym or len(word) < 3 or not word.isalpha():
            return word
        suggestions = self._sym.lookup(
            word.lower(),
            Verbosity.CLOSEST,
            max_edit_distance=self.max_edit_distance,
            include_unknown=True,
        )
        if not suggestions:
            return word
        return suggestions[0].term

    def correct(self, text: str) -> str:
        tokens = text.split()
        return " ".join(self.correct_word(t) for t in tokens)
