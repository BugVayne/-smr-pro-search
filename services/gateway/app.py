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
    log.info("━━━ SEARCH REQUEST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  query   : «%s»", req.query)
    if req.context:
        log.info("  context : %s", req.context)
    resp = run_pipeline(req)
    t = resp.timings_ms
    total = sum(t.values())
    log.info("━━━ SEARCH RESULT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  results : %d items, %d kits, %d ptm_groups, type=%s",
             len(resp.items), len(resp.kits), len(resp.ptm_groups), resp.query_type)
    log.info("  timings : pre=%.0fms ret=%.0fms rank=%.0fms post=%.0fms | total=%.0fms",
             t.get("preprocess", 0), t.get("retrieve", 0),
             t.get("rank", 0), t.get("postprocess", 0), total)
    if resp.items:
        top3 = ", ".join(f"{it.obosn}({it.score:.4f})" for it in resp.items[:3])
        log.info("  top3    : %s", top3)
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
