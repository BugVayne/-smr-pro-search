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

from services.postprocessing.kit_builder import build_kits, build_ptm_groups, enrich_with_related

log = get_logger("postprocessing")
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "postprocessing"})


@app.post("/postprocess")
def postprocess():
    payload = PostprocessRequest(**request.get_json(force=True))

    log.info("POSTPROCESS type=%s", payload.query_type)
    log.info("  items     : %d", len(payload.items))

    if payload.query_type == "atomic":
        items = enrich_with_related(payload.items)
        kits = []
        ptm_groups = []
        enriched = sum(1 for it in items if it.related)
        log.info("  enriched  : %d/%d items have related расценки", enriched, len(items))
    else:
        items = payload.items
        kits = build_kits(items, top_seeds=8)
        ptm_groups = build_ptm_groups(payload.matched_ptms)
        if kits:
            seeds = ", ".join(k.seed_obosn for k in kits)
            log.info("  kits      : %d built | seeds: %s", len(kits), seeds)
        else:
            log.info("  kits      : 0 (no association rules matched)")
        if ptm_groups:
            names = ", ".join(g.ptm_naim for g in ptm_groups)
            log.info("  ptm_groups: %d | PTMs: %s", len(ptm_groups), names)
        else:
            log.info("  ptm_groups: 0 (no PTM match)")

    return jsonify(PostprocessResponse(items=items, kits=kits, ptm_groups=ptm_groups).model_dump())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORTS["postprocessing"], debug=False)
