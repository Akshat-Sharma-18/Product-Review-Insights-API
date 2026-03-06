"""
demo_store.py
=============
In-memory store for Demo Product Mode.

Keeps demo products and their reviews entirely in memory.
All data is lost on server restart — by design.

This module is intentionally decoupled from the real dataset so that
demo operations can never corrupt REVIEWS_DB or DataLoader state.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_REVIEWS_PER_DEMO_PRODUCT: int = 100
MAX_DEMO_PRODUCTS:            int = 50    # guard against unbounded memory growth
MAX_REVIEW_TEXT_LENGTH:       int = 5_000
MAX_SUMMARY_LENGTH:           int = 500
MAX_PRODUCT_ID_LENGTH:        int = 100

# ── In-memory store ───────────────────────────────────────────────────────────
# Structure:
#   DEMO_PRODUCTS = {
#       "<product_id>": [
#           {
#               "review_text": str,
#               "summary":     str,
#               "rating":      float | None,
#           },
#           ...
#       ]
#   }

DEMO_PRODUCTS: dict[str, list[dict]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_pid(raw: str) -> str:
    """Strip and upper-case a product ID, matching the convention in data_loader."""
    return raw.strip().upper()


def _parse_rating(raw) -> Optional[float]:
    """Parse and validate a 1–5 star rating. Returns None if invalid."""
    if raw is None:
        return None
    try:
        r = float(raw)
        return r if 1.0 <= r <= 5.0 else None
    except (ValueError, TypeError):
        return None


def _validate_product_id(raw) -> tuple[Optional[str], Optional[str]]:
    """
    Validate and normalise a product_id.

    Returns (normalised_pid, error_message).
    error_message is None when the ID is valid.
    """
    if not raw or not str(raw).strip():
        return None, "product_id is required and must not be empty."
    pid = str(raw).strip()
    if len(pid) > MAX_PRODUCT_ID_LENGTH:
        return None, f"product_id must be {MAX_PRODUCT_ID_LENGTH} characters or fewer."
    # Reject IDs that could look like path traversal or injection attempts
    if not re.match(r"^[\w\-. ]+$", pid):
        return None, "product_id contains invalid characters."
    return _normalise_pid(pid), None


def _validate_review(review_text, summary, rating) -> Optional[str]:
    """
    Validate review fields.

    Returns an error string, or None when all fields are acceptable.
    """
    if not review_text or not str(review_text).strip():
        return "review_text is required and must not be empty."
    if len(str(review_text).strip()) < 5:
        return "review_text is too short (minimum 5 characters)."
    if len(str(review_text)) > MAX_REVIEW_TEXT_LENGTH:
        return f"review_text must be {MAX_REVIEW_TEXT_LENGTH} characters or fewer."
    if summary and len(str(summary)) > MAX_SUMMARY_LENGTH:
        return f"summary must be {MAX_SUMMARY_LENGTH} characters or fewer."
    if rating is not None and _parse_rating(rating) is None:
        return "rating must be a number between 1 and 5."
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def create_demo_product(raw_pid: str) -> tuple[dict, bool]:
    """
    Create a new demo product slot in memory.

    Returns (result_dict, already_existed).
    result_dict always contains "product_id".
    Raises ValueError when the product_id is invalid or the store is full.
    """
    pid, err = _validate_product_id(raw_pid)
    if err:
        raise ValueError(err)

    if pid in DEMO_PRODUCTS:
        logger.info("Demo product already exists: %s", pid)
        return {"product_id": pid, "review_count": len(DEMO_PRODUCTS[pid])}, True

    if len(DEMO_PRODUCTS) >= MAX_DEMO_PRODUCTS:
        raise ValueError(
            f"Maximum number of demo products ({MAX_DEMO_PRODUCTS}) reached. "
            "Delete an existing demo product before creating a new one."
        )

    DEMO_PRODUCTS[pid] = []
    logger.info("Demo product created: %s", pid)
    return {"product_id": pid, "review_count": 0}, False


def add_demo_review(
    raw_pid: str,
    review_text: str,
    summary: str = "",
    rating=None,
) -> dict:
    """
    Append a review to an existing demo product.

    Returns {"product_id": str, "review_count": int}.
    Raises ValueError on validation failure or missing product.
    Raises KeyError when the product_id does not exist.
    """
    pid, err = _validate_product_id(raw_pid)
    if err:
        raise ValueError(err)

    if pid not in DEMO_PRODUCTS:
        raise KeyError(f"Demo product '{pid}' not found. Create it first.")

    err = _validate_review(review_text, summary, rating)
    if err:
        raise ValueError(err)

    reviews = DEMO_PRODUCTS[pid]
    if len(reviews) >= MAX_REVIEWS_PER_DEMO_PRODUCT:
        raise ValueError(
            f"Demo product '{pid}' already has the maximum "
            f"{MAX_REVIEWS_PER_DEMO_PRODUCT} reviews."
        )

    review = {
        "review_text": str(review_text).strip()[:MAX_REVIEW_TEXT_LENGTH],
        "summary":     str(summary or "").strip()[:MAX_SUMMARY_LENGTH],
        "rating":      _parse_rating(rating),
    }
    reviews.append(review)
    count = len(reviews)
    logger.info("Demo review added for product %s (total: %d).", pid, count)
    return {"product_id": pid, "review_count": count}


def get_demo_reviews(raw_pid: str) -> list[dict]:
    """
    Retrieve all reviews for a demo product.

    Raises ValueError on bad product_id.
    Raises KeyError when the product does not exist.
    """
    pid, err = _validate_product_id(raw_pid)
    if err:
        raise ValueError(err)

    if pid not in DEMO_PRODUCTS:
        raise KeyError(f"Demo product '{pid}' not found.")

    return DEMO_PRODUCTS[pid]


def delete_demo_product(raw_pid: str) -> str:
    """
    Remove a demo product and all its reviews from memory.

    Returns the normalised product_id.
    Raises ValueError on bad product_id.
    Raises KeyError when the product does not exist.
    """
    pid, err = _validate_product_id(raw_pid)
    if err:
        raise ValueError(err)

    if pid not in DEMO_PRODUCTS:
        raise KeyError(f"Demo product '{pid}' not found.")

    del DEMO_PRODUCTS[pid]
    logger.info("Demo product deleted: %s", pid)
    return pid


def list_demo_products() -> list[dict]:
    """Return a summary list of all current demo products."""
    return [
        {"product_id": pid, "review_count": len(reviews)}
        for pid, reviews in DEMO_PRODUCTS.items()
    ]
