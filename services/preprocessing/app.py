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

from services.preprocessing.classifier import QueryClassifier
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
    log.info("preprocess: %r", payload.query)

    corrected = _speller.correct(payload.query)
    lemmas = lemmatize(corrected)
    expanded = _expander.expand(lemmas)
    qtype = _classifier.predict(payload.query)

    resp = PreprocessResponse(
        original=payload.query,
        corrected=corrected,
        lemmas=lemmas,
        expanded_terms=expanded,
        query_type=qtype,
    )
    return jsonify(resp.model_dump())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORTS["preprocessing"], debug=False)
