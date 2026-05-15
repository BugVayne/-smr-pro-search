"""Layer 4: postprocessing service.

Forms technological kits via FP-Growth association rules and returns
the final structured response.

    POST /postprocess  PostprocessRequest -> PostprocessResponse
"""
from __future__ import annotations

from flask import Flask, jsonify, request

from common.config import PORTS
from common.logging_config import get_logger
from common.models import PostprocessRequest, PostprocessResponse

from services.postprocessing.kit_builder import build_kits

log = get_logger("postprocessing")
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "postprocessing"})


@app.post("/postprocess")
def postprocess():
    payload = PostprocessRequest(**request.get_json(force=True))
    log.info("postprocess: %d items, type=%s", len(payload.items), payload.query_type)

    # Kits are always useful, but for aggregated queries we expose them
    # more prominently (deeper search across top seeds).
    top_seeds = 8 if payload.query_type == "aggregated" else 5
    kits = build_kits(payload.items, top_seeds=top_seeds)

    return jsonify(PostprocessResponse(items=payload.items, kits=kits).model_dump())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORTS["postprocessing"], debug=False)
