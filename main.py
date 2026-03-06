"""
main.py  –  Product Review Insights API  (Flask)
=================================================
Loads  Reviews1.csv   (CSV  – ProductId / Score / Text)
  AND  Reviews2.json  (JSONL – asin / overall / reviewText, one object per line)

Endpoints (original):
  GET  /api/v1/health
  GET  /api/v1/products[?search=X&page=1&limit=100]
  GET  /api/v1/insights?product_id=<id>
  POST /api/v1/insights   body: {"product_id": "<id>"}

Endpoints (Demo Product Mode):
  POST   /api/v1/demo-product                         – create demo product
  POST   /api/v1/demo-review                          – add review to demo product
  GET    /api/v1/demo-insights?product_id=<id>        – analyse demo product
  DELETE /api/v1/demo-product?product_id=<id>         – remove demo product
  GET    /api/v1/demo-products                        – list all demo products
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from analyzer import ReviewAnalyzer
from data_loader import get_product_reviews, load_all
from demo_store import (
    add_demo_review,
    create_demo_product,
    delete_demo_product,
    get_demo_reviews,
    list_demo_products,
)


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "api.log"),
    ],
)
logger = logging.getLogger("api")

# ── Dataset paths ─────────────────────────────────────────────────────────────
BASE = Path(__file__).parent / "data"

CSV1  = Path(os.getenv("REVIEWS_CSV",  BASE / "Reviews1.csv"))
JSON2 = Path(os.getenv("REVIEWS_JSON", BASE / "Reviews2.json"))

_to_load = [p for p in [CSV1, JSON2] if p.exists()]

REVIEWS_DB: dict = {}
if _to_load:
    logger.info("Loading files: %s", [p.name for p in _to_load])
    REVIEWS_DB = load_all(_to_load)
else:
    logger.error(
        "No data files found. Expected:\n  %s\n  %s\n"
        "Place your files there or set REVIEWS_CSV / REVIEWS_JSON env vars.",
        CSV1, JSON2,
    )

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.before_request
def _start_timer():
    request._start = time.perf_counter()


@app.after_request
def _add_timing(response):
    ms = (time.perf_counter() - getattr(request, "_start", time.perf_counter())) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(ms, 2))
    return response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _err(msg: str, detail: str | None = None, code: int = 400):
    body = {"error": msg}
    if detail:
        body["detail"] = detail
    return jsonify(body), code


def _validate_pid(raw) -> tuple[str | None, tuple | None]:
    if not raw or not str(raw).strip():
        return None, _err("product_id is required and must not be empty.")
    pid = str(raw).strip()
    if len(pid) > 100:
        return None, _err("product_id must be 100 characters or fewer.")
    return pid.upper(), None


def _analyse(product_id_raw):
    pid, err = _validate_pid(product_id_raw)
    if err:
        return err

    if not REVIEWS_DB:
        return _err(
            "No dataset loaded. Place Reviews1.csv and Reviews2.json in the data/ folder.",
            code=503,
        )

    if pid not in REVIEWS_DB:
        logger.warning("Product not found: %s", pid)
        return _err(f"Product '{pid}' not found in dataset.", code=404)

    reviews = get_product_reviews(REVIEWS_DB, pid)
    logger.info("Analysing %d reviews for product_id=%s", len(reviews), pid)

    result = ReviewAnalyzer(reviews).analyze()
    return jsonify({
        "product_id":   pid,
        "review_count": result["review_count"],
        "top_aspects":  result["top_aspects"],
        "pros":         result["pros"],
        "cons":         result["cons"],
        "summary":      result["summary"],
        "confidence":   result["confidence"],
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/v1/health", methods=["GET"])
def health():
    sources = [p.name for p in [CSV1, JSON2] if p.exists()]
    return jsonify({
        "status":           "ok" if REVIEWS_DB else "degraded",
        "files_loaded":     sources,
        "products_loaded":  len(REVIEWS_DB),
        "total_reviews":    sum(len(v) for v in REVIEWS_DB.values()),
    })


@app.route("/api/v1/products", methods=["GET"])
def products():
    all_pids = sorted(REVIEWS_DB.keys())

    search = request.args.get("search", "").upper()
    if search:
        all_pids = [p for p in all_pids if search in p]

    try:
        page  = max(1, int(request.args.get("page", 1)))
        limit = min(500, max(1, int(request.args.get("limit", 100))))
    except ValueError:
        return _err("page and limit must be integers.")

    start = (page - 1) * limit
    return jsonify({
        "products": all_pids[start: start + limit],
        "total":    len(all_pids),
        "page":     page,
        "limit":    limit,
        "pages":    -(-len(all_pids) // limit),
    })


@app.route("/api/v1/insights", methods=["POST"])
def post_insights():
    """Body (JSON): {"product_id": "B001E4KFA3"}"""
    if not request.is_json:
        return _err("Content-Type must be application/json.")
    body = request.get_json(silent=True) or {}
    return _analyse(body.get("product_id", ""))


@app.route("/api/v1/insights", methods=["GET"])
def get_insights():
    """GET /api/v1/insights?product_id=B001E4KFA3"""
    return _analyse(request.args.get("product_id", ""))

@app.route("/")
def index():
    return send_from_directory(".", "index.html")




# ── Routes (Demo Product Mode) ────────────────────────────────────────────────

@app.route("/api/v1/demo-product", methods=["POST"])
def demo_create_product():
    """
    POST /api/v1/demo-product
    Body: {"product_id": "demo_phone_1"}

    Creates an empty demo product slot in memory.
    Idempotent: returns 200 with existing status if the product already exists.
    """
    if not request.is_json:
        return _err("Content-Type must be application/json.")

    body = request.get_json(silent=True) or {}
    raw_pid = body.get("product_id", "")

    try:
        result, already_existed = create_demo_product(raw_pid)
    except ValueError as exc:
        return _err(str(exc))

    if already_existed:
        return jsonify({
            "message":      "Demo product already exists.",
            "product_id":   result["product_id"],
            "review_count": result["review_count"],
        }), 200

    return jsonify({
        "message":    "Demo product created.",
        "product_id": result["product_id"],
    }), 201


@app.route("/api/v1/demo-review", methods=["POST"])
def demo_add_review():
    """
    POST /api/v1/demo-review
    Body: {
        "product_id":  "demo_phone_1",
        "review_text": "Battery lasts only two weeks with heavy usage",
        "summary":     "",        (optional)
        "rating":      2          (optional, 1-5)
    }

    Appends a validated review to an existing demo product.
    """
    if not request.is_json:
        return _err("Content-Type must be application/json.")

    body = request.get_json(silent=True) or {}

    raw_pid     = body.get("product_id", "")
    review_text = body.get("review_text", "")
    summary     = body.get("summary", "")
    rating      = body.get("rating")

    try:
        result = add_demo_review(raw_pid, review_text, summary, rating)
    except KeyError as exc:
        return _err(str(exc), code=404)
    except ValueError as exc:
        return _err(str(exc))

    return jsonify({
        "message":      "Review added.",
        "review_count": result["review_count"],
    }), 201


@app.route("/api/v1/demo-insights", methods=["GET"])
def demo_get_insights():
    """
    GET /api/v1/demo-insights?product_id=demo_phone_1

    Runs the full ReviewAnalyzer pipeline on in-memory demo reviews.
    Response shape is identical to /api/v1/insights.
    """
    raw_pid = request.args.get("product_id", "")

    try:
        reviews = get_demo_reviews(raw_pid)
    except KeyError as exc:
        return _err(str(exc), code=404)
    except ValueError as exc:
        return _err(str(exc))

    logger.info(
        "Analysing demo product '%s' with %d review(s).",
        raw_pid.strip().upper(), len(reviews),
    )

    result = ReviewAnalyzer(reviews).analyze()

    return jsonify({
        "product_id":         raw_pid.strip().upper(),
        "review_count":       result["review_count"],
        "top_aspects":        result["top_aspects"],
        "pros":               result["pros"],
        "cons":               result["cons"],
        "summary":            result["summary"],
        "confidence":         result["confidence"],
        "discovered_aspects": result.get("discovered_aspects", []),
    })


@app.route("/api/v1/demo-product", methods=["DELETE"])
def demo_delete_product():
    """
    DELETE /api/v1/demo-product?product_id=demo_phone_1

    Removes the demo product and all its reviews from memory.
    """
    raw_pid = request.args.get("product_id", "")

    try:
        pid = delete_demo_product(raw_pid)
    except KeyError as exc:
        return _err(str(exc), code=404)
    except ValueError as exc:
        return _err(str(exc))

    return jsonify({
        "message":    "Demo product deleted.",
        "product_id": pid,
    })


@app.route("/api/v1/demo-products", methods=["GET"])
def demo_list_products():
    """
    GET /api/v1/demo-products

    Returns a summary list of all demo products currently held in memory.
    Useful for debugging and admin visibility.
    """
    items = list_demo_products()
    return jsonify({
        "demo_products": items,
        "total":         len(items),
    })

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info("Starting Product Review Insights API on port %d ...", port)
    app.run(host="0.0.0.0", port=port, debug=False)
