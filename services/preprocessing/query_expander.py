"""Query expansion.

Two sources are combined:

* a curated thesaurus (hand-edited mapping from professional slang to
  normative terms), which is small but high-precision;
* a FastText model trained on the corpus of rascenka names, which can
  propose lexically and semantically similar terms automatically.

Either source can be missing — the code degrades gracefully and simply
returns fewer expansions.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from common.config import FASTTEXT_MODEL_PATH
from common.logging_config import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Curated thesaurus
# ---------------------------------------------------------------------------

# Seed thesaurus; in production this would live in a JSON file or DB and
# be edited by domain experts.
CURATED_THESAURUS = {
    "тёплый": ["теплый", "обогреваемый"],
    "теплый": ["обогреваемый", "отапливаемый"],
    "пол": ["напольный", "стяжка", "покрытие"],
    "кровля": ["крыша", "кровельный"],
    "пирог": ["слой", "конструкция"],
    "стена": ["стенной", "перегородка"],
    "покраска": ["окраска", "окрашивание"],
    "штукатурка": ["оштукатуривание"],
    "плитка": ["облицовка"],
    "горелка": ["котёл", "котел"],
    "стяжка": ["заливка", "выравнивание"],
}


def thesaurus_expand(lemma: str) -> List[str]:
    return CURATED_THESAURUS.get(lemma, [])


# ---------------------------------------------------------------------------
# FastText expansion
# ---------------------------------------------------------------------------

class FastTextExpander:
    def __init__(self, model_path: Optional[Path] = None, topn: int = 3, min_sim: float = 0.65):
        self.model = None
        self.topn = topn
        self.min_sim = min_sim

        model_path = model_path or FASTTEXT_MODEL_PATH
        if not Path(model_path).exists():
            log.warning(
                "FastText model %s not found – semantic expansion disabled",
                model_path,
            )
            return

        try:
            from gensim.models import FastText
            self.model = FastText.load(str(model_path))
            log.info("Loaded FastText model from %s", model_path)
        except Exception as exc:
            log.warning("Could not load FastText: %s", exc)

    def expand_token(self, token: str) -> List[str]:
        if self.model is None:
            return []
        try:
            neighbours = self.model.wv.most_similar(token, topn=self.topn)
        except KeyError:
            return []
        return [w for w, sim in neighbours if sim >= self.min_sim]


# ---------------------------------------------------------------------------
# Combined expander
# ---------------------------------------------------------------------------

class QueryExpander:
    def __init__(self):
        self._ft = FastTextExpander()

    def expand(self, lemmas: List[str]) -> List[str]:
        seen = set(lemmas)
        out: List[str] = []
        for lemma in lemmas:
            for term in thesaurus_expand(lemma) + self._ft.expand_token(lemma):
                if term and term not in seen:
                    seen.add(term)
                    out.append(term)
        return out
