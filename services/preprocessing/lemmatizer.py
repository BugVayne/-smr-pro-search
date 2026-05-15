"""Lemmatization for Russian text.

Uses pymorphy3 (a fork of pymorphy2 with active maintenance). For
construction-domain stop words we keep a small list that is *only*
filtered when it is not the sole content word of the query, to preserve
discriminative power in single-word queries like "монтаж".
"""
from __future__ import annotations

import re
from typing import List

from common.logging_config import get_logger

log = get_logger(__name__)

try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
except ImportError:
    log.warning("pymorphy3 not installed – using identity lemmatizer")
    _morph = None

# Generic construction terms that are too common to be discriminative.
DOMAIN_STOPWORDS = {
    "устройство", "монтаж", "прокладка", "работа", "работы",
    "выполнение", "проведение", "производство",
}

# Common Russian function words to drop unconditionally.
RUSSIAN_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был",
    "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там",
}


_TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9-]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def lemmatize_token(token: str) -> str:
    if _morph is None:
        return token
    parses = _morph.parse(token)
    if not parses:
        return token
    return parses[0].normal_form


def lemmatize(text: str) -> List[str]:
    """Tokenize, lemmatize, and filter stop words.

    Domain stop words are kept if they are the only content word; this
    prevents queries like "монтаж" from becoming empty.
    """
    tokens = tokenize(text)
    lemmas = [lemmatize_token(t) for t in tokens]
    lemmas = [l for l in lemmas if l not in RUSSIAN_STOPWORDS]

    content = [l for l in lemmas if l not in DOMAIN_STOPWORDS]
    return content if content else lemmas
