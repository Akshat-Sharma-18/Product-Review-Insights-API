"""
data_loader.py
==============
Loads and normalises reviews from TWO different file formats:

Reviews1.csv  (Amazon Food Reviews - CSV)
  Columns: Id, ProductId, UserId, ProfileName,
           HelpfulnessNumerator, HelpfulnessDenominator,
           Score, Time, Summary, Text

Reviews2.json  (Amazon Electronics - JSONL, one object per line)
  Fields: asin, reviewerID, reviewerName, helpful,
          overall, reviewText, summary,
          reviewTime, unixReviewTime

Both are normalised into a common internal dict:
  {
    "product_id"  : str,
    "review_text" : str,
    "summary"     : str,
    "rating"      : float | None,
    "review_date" : str,
    "reviewer_id" : str,
    "source_file" : str,   # "reviews1" | "reviews2"
  }
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags
    text = re.sub(r"https?://\S+", "", text)        # strip URLs
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_rating(raw) -> float | None:
    try:
        r = float(str(raw).strip())
        return r if 1.0 <= r <= 5.0 else None
    except (ValueError, TypeError):
        return None


# ── Reviews1.csv loader ───────────────────────────────────────────────────────

def load_csv(csv_path: str | Path) -> dict[str, list[dict]]:
    """
    Load Reviews1.csv
    Required columns: ProductId, Score, Text
    Optional columns: UserId, Summary, Time
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    reviews_by_product: dict[str, list[dict]] = {}
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = {c.lower() for c in (reader.fieldnames or [])}

        # Validate required columns
        required = {"productid", "score", "text"}
        if not required.issubset(fields):
            missing = required - fields
            raise ValueError(
                f"{csv_path.name}: missing required columns {missing}. "
                f"Found: {list(reader.fieldnames)}"
            )

        logger.info("  %s - CSV schema detected (Reviews1 style)", csv_path.name)

        for row in reader:
            pid  = (row.get("ProductId") or row.get("productid") or "").strip()
            text = (row.get("Text")      or row.get("text")      or "").strip()
            if not pid or not text or len(text) < 15:
                skipped += 1
                continue

            reviews_by_product.setdefault(pid.upper(), []).append({
                "product_id":  pid.upper(),
                "reviewer_id": (row.get("UserId") or row.get("userid") or "").strip(),
                "rating":      _parse_rating(row.get("Score") or row.get("score")),
                "review_date": (row.get("Time")    or row.get("time")    or "").strip(),
                "summary":     _clean(row.get("Summary") or row.get("summary") or ""),
                "review_text": _clean(text),
                "source_file": "reviews1",
            })

    total = sum(len(v) for v in reviews_by_product.values())
    logger.info("  %s - %d products, %d reviews loaded (%d skipped).",
                csv_path.name, len(reviews_by_product), total, skipped)
    return reviews_by_product


# ── Reviews2.json (JSONL) loader ──────────────────────────────────────────────

def load_jsonl(json_path: str | Path, max_reviews_per_product: int = 500) -> dict[str, list[dict]]:
    """
    Load Reviews2.json - JSONL format (one JSON object per line).
    Memory-efficient: caps reviews per product and truncates long texts.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    reviews_by_product: dict[str, list[dict]] = {}
    # Track count per product to cap memory usage
    product_counts: dict[str, int] = {}
    skipped = 0
    line_num = 0

    with open(json_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_num += 1

            if line_num % 200000 == 0:
                logger.info("  %s - reading line %d (%d products so far)...",
                            json_path.name, line_num, len(reviews_by_product))

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            pid  = str(obj.get("asin") or "").strip().upper()
            text = str(obj.get("reviewText") or "").strip()

            if not pid or not text or len(text) < 15:
                skipped += 1
                continue

            # Cap reviews per product to save memory
            if product_counts.get(pid, 0) >= max_reviews_per_product:
                continue

            # Truncate very long reviews to save memory (keep first 1000 chars)
            if len(text) > 1000:
                text = text[:1000]

            reviews_by_product.setdefault(pid, []).append({
                "product_id":  pid,
                "reviewer_id": str(obj.get("reviewerID") or "").strip(),
                "rating":      _parse_rating(obj.get("overall")),
                "review_date": str(obj.get("reviewTime") or "").strip(),
                "summary":     _clean(str(obj.get("summary") or ""))[:200],
                "review_text": _clean(text),
                "source_file": "reviews2",
            })
            product_counts[pid] = product_counts.get(pid, 0) + 1

    total = sum(len(v) for v in reviews_by_product.values())
    logger.info("  %s - %d products, %d reviews loaded (%d skipped).",
                json_path.name, len(reviews_by_product), total, skipped)
    return reviews_by_product


# ── Merge helper ──────────────────────────────────────────────────────────────

def load_all(files: list[str | Path]) -> dict[str, list[dict]]:
    """
    Load any mix of .csv and .json/.jsonl files, merge and de-duplicate.
    De-duplication key: (reviewer_id, first 80 chars of review_text)
    """
    merged: dict[str, list[dict]] = {}

    for path in files:
        path = Path(path)
        if not path.exists():
            logger.warning("File not found, skipping: %s", path)
            continue
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                db = load_csv(path)
            elif suffix in (".json", ".jsonl"):
                db = load_jsonl(path)
            else:
                logger.warning("Unknown file type '%s', skipping: %s", suffix, path)
                continue
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            continue

        for pid, reviews in db.items():
            merged.setdefault(pid, []).extend(reviews)

    # De-duplicate
    deduped: dict[str, list[dict]] = {}
    for pid, reviews in merged.items():
        seen: set[tuple] = set()
        unique = []
        for r in reviews:
            key = (r["reviewer_id"], r["review_text"][:80])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        deduped[pid] = unique

    total = sum(len(v) for v in deduped.values())
    logger.info("Merged total: %d products, %d unique reviews.", len(deduped), total)
    return deduped


# ── Lookup ────────────────────────────────────────────────────────────────────

def get_product_reviews(db: dict[str, list[dict]], product_id: str) -> list[dict]:
    """Return reviews for product_id (case-insensitive). Empty list if not found."""
    return db.get(product_id.strip().upper(), [])