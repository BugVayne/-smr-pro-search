"""Gateway service.

Public entry point for the search system. Wraps the full pipeline and
also accepts user feedback events.

    POST /search    SearchRequest -> SearchResponse
    POST /feedback  { query, obosn, label }
"""
from __future__ import annotations

from flask import Flask, jsonify, request

from common.config import PORTS
from common.db import insert_feedback
from common.logging_config import get_logger
from common.models import SearchRequest

from services.gateway.pipeline import run_pipeline

log = get_logger("gateway")
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "gateway"})


@app.post("/search")
def search():
    body = request.get_json(force=True)
    req = SearchRequest(**body)
    log.info("search: %r", req.query)
    resp = run_pipeline(req)
    return jsonify(resp.model_dump())


@app.post("/feedback")
def feedback():
    body = request.get_json(force=True)
    insert_feedback(
        query=body["query"],
        obosn=body["obosn"],
        label=int(body.get("label", 2)),
    )
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORTS["gateway"], debug=False)
