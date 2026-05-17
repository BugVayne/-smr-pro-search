"""Layer 1: query preprocessing service.

Exposes:

    POST /preprocess  { "query": "..." } -> PreprocessResponse
    GET  /health
"""
from __future__ import annotations

from flask import Flask, jsonify, request

from common.config import PORTS
from common.logging_config import get_logger
from common.models import PreprocessRequest, PreprocessResponse

from services.preprocessing.classifier import CODE_RE, QueryClassifier
from services.preprocessing.lemmatizer import lemmatize
from services.preprocessing.query_expander import QueryExpander
from services.preprocessing.spell_correction import SpellCorrector

log = get_logger("preprocessing")
app = Flask(__name__)

# Lazily instantiated singletons (loaded once per process).
_speller = SpellCorrector()
_expander = QueryExpander()
_classifier = QueryClassifier()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "preprocessing"})


@app.post("/preprocess")
def preprocess():
    payload = PreprocessRequest(**request.get_json(force=True))

    corrected = _speller.correct(payload.query)
    lemmas = lemmatize(corrected)
    expanded = _expander.expand(lemmas)
    qtype = _classifier.predict(payload.query)

    m = CODE_RE.search(payload.query)
    extracted_code = m.group(0).upper() if m else None

    spell_note = "unchanged" if corrected == payload.query else f"→ «{corrected}»"
    new_terms = [t for t in expanded if t not in lemmas]
    log.info("PREPROCESS «%s»", payload.query)
    log.info("  spell     : %s", spell_note)
    log.info("  lemmas(%d) : %s", len(lemmas), lemmas)
    log.info("  expanded(%d): +%s", len(new_terms), new_terms)
    log.info("  type      : %s", qtype)
    if extracted_code:
        log.info("  code      : %s (direct lookup)", extracted_code)

    resp = PreprocessResponse(
        original=payload.query,
        corrected=corrected,
        lemmas=lemmas,
        expanded_terms=expanded,
        query_type=qtype,
        extracted_code=extracted_code,
    )
    return jsonify(resp.model_dump())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORTS["preprocessing"], debug=False)
