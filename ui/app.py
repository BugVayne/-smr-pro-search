"""Flask UI – a small web front-end over the gateway service.

Routes:
    GET  /         search page
    POST /search   submit a query, render results
    POST /feedback record click feedback (AJAX)
"""
from __future__ import annotations

import requests
from flask import Flask, jsonify, render_template, request

from common.config import GATEWAY_URL, PORTS
from common.logging_config import get_logger

log = get_logger("ui")
app = Flask(__name__)


@app.get("/")
def index():
    return render_template("search.html")


@app.post("/search")
def search():
    query = request.form.get("query", "").strip()
    if not query:
        return render_template("search.html", error="Введите запрос")

    try:
        r = requests.post(
            f"{GATEWAY_URL}/search",
            json={"query": query, "context": {}},
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()
    except requests.RequestException as exc:
        log.warning("Gateway call failed: %s", exc)
        return render_template(
            "search.html", error=f"Сервис недоступен: {exc}", query=query
        )

    return render_template("results.html", query=query, result=result)


@app.post("/feedback")
def feedback():
    body = request.get_json(force=True)
    try:
        requests.post(f"{GATEWAY_URL}/feedback", json=body, timeout=5)
    except requests.RequestException as exc:
        log.warning("Feedback recording failed: %s", exc)
        return jsonify({"status": "error", "reason": str(exc)}), 502
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORTS["ui"], debug=True)
