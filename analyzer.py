"""
analyzer.py
===========
Hybrid NLP engine for Product Review Insights.

Improvements over baseline:
  1. SENTIMENT ACCURACY   – DeBERTa ABSA (aspect-aware) → DistilBERT → lexicon
                            Confidence-weighted scoring.
  2. SMARTER ASPECTS      – SentenceTransformer cosine similarity + keyword boost.
                            Adaptive threshold per sentence length.
  3. RICHER EVIDENCE      – Best sentence chosen by composite score:
                            |sentiment| × informativeness (length + diversity).
  4. CONFIDENCE SCORING   – Three-axis: volume · rating agreement · helpfulness.
  5. BATCHED INFERENCE    – All model calls batched in one pass per product.
                            Typically 5-10x faster than one-at-a-time.

Pipeline:
  1. Text preprocessing      (handled in data_loader)
  2. Collect all sentences across all reviews for a product
  3. Batch aspect detection  – one SentenceTransformer encode call for all
  4. Batch sentiment         – one ABSA / DistilBERT call for all pairs
  5. Evidence selection      – composite informativeness score
  6. Pros / cons             – only when evidence exists
  7. Confidence scoring      – volume + agreement + helpfulness
  8. Topic discovery         – BERTopic (optional, non-blocking)
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Aspect taxonomy ────────────────────────────────────────────────────────────
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "battery life":    ["battery", "charge", "charging", "last", "lasts", "lasting", "power"],
    "performance":     ["performance", "speed", "fast", "slow", "snappy", "lag", "responsive",
                        "smooth", "processing", "processor", "cpu"],
    "display/screen":  ["screen", "display", "resolution", "brightness", "colors", "vivid",
                        "crisp", "clear", "picture", "monitor"],
    "build quality":   ["build", "quality", "material", "plastic", "premium", "cheap", "solid",
                        "sturdy", "durable", "durability", "flimsy"],
    "design":          ["design", "look", "looks", "aesthetic", "style", "slim", "thin",
                        "compact", "sleek"],
    "keyboard":        ["keyboard", "keys", "typing", "trackpad", "touchpad"],
    "comfort":         ["comfort", "comfortable", "uncomfortable", "fit", "tight", "heavy",
                        "cushion", "ergonomic"],
    "sound quality":   ["sound", "audio", "bass", "treble", "highs", "noise", "cancellation",
                        "immersive", "rich", "volume", "speaker", "speakers"],
    "camera":          ["camera", "photo", "photos", "shot", "shots", "image", "pictures",
                        "megapixel", "low light", "optical"],
    "price/value":     ["price", "value", "worth", "expensive", "cheap", "cost", "affordable",
                        "overpriced", "money"],
    "shipping":        ["shipping", "delivery", "arrived", "packaging", "package", "shipped",
                        "dispatch"],
    "customer service":["service", "support", "seller", "return", "refund", "warranty",
                        "customer service", "contact"],
    "taste/flavor":    ["taste", "flavor", "flavour", "delicious", "yummy", "sweet", "salty",
                        "bitter", "fresh", "stale", "savory"],
    "smell/scent":     ["smell", "scent", "odor", "fragrance", "aroma"],
    "size/fit":        ["size", "sizing", "fit", "fits", "small", "large", "medium", "tight",
                        "loose", "length", "width"],
    "ease of use":     ["easy", "simple", "straightforward", "intuitive", "setup", "install",
                        "user-friendly", "complicated", "confusing"],
    "durability":      ["durable", "durability", "lasted", "broke", "broken", "long-lasting",
                        "wear", "worn"],
    "gps/navigation":  ["gps", "navigation", "route", "map", "maps", "directions", "location",
                        "signal", "accuracy", "accurate"],
    "food quality":    ["food", "ingredient", "ingredients", "organic", "natural", "healthy",
                        "nutrition", "diet", "calorie", "protein"],
}

# ── Sentiment lexicon (rule-based fallback) ───────────────────────────────────
POSITIVE_WORDS = {
    "amazing", "great", "excellent", "outstanding", "exceptional", "fantastic",
    "love", "incredible", "phenomenal", "brilliant", "stunning", "gorgeous",
    "crisp", "clear", "vivid", "immersive", "rich", "powerful", "smooth",
    "impressive", "good", "best", "perfect", "superb", "decent", "solid",
    "premium", "nice", "fast", "snappy", "responsive", "beautiful", "wonderful",
    "delicious", "yummy", "fresh", "accurate", "reliable", "recommended",
    "satisfied", "happy", "pleased", "quality", "convenient", "helpful",
    "effective", "efficient", "durable", "comfortable", "easy",
    "intuitive", "affordable", "worth", "value", "healthy", "natural",
    "top-notch", "crystal", "deep", "long", "full", "highly", "super",
}
NEGATIVE_WORDS = {
    "poor", "bad", "terrible", "awful", "lacking", "mediocre", "disappointing",
    "cheap", "uncomfortable", "heavy", "tight", "mushy", "below", "struggles",
    "hurt", "washed", "weak", "slow", "lag", "broken", "broke", "defective",
    "useless", "waste", "horrible", "disgusting", "stale", "rotten", "expired",
    "confusing", "complicated", "overpriced", "expensive", "flimsy",
    "unreliable", "freezing", "froze", "crash", "crashes", "failed", "failure",
    "worst", "never", "avoid", "regret", "disappointed", "unhappy", "returned",
    "return", "refund", "problem", "issue", "issues", "wrong", "incorrect",
}
NEGATION_WORDS = {
    "not", "no", "never", "without", "hardly", "barely",
    "doesn't", "didn't", "won't", "wasn't", "isn't", "aren't", "don't",
}

# ── Model loading ─────────────────────────────────────────────────────────────

_absa_pipeline  = None
_absa_available = False
try:
    from transformers import pipeline as hf_pipeline
    logger.info("Loading ABSA model (yangheng/deberta-v3-base-absa-v1.1) ...")
    _absa_pipeline = hf_pipeline(
        "text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1",
        tokenizer="yangheng/deberta-v3-base-absa-v1.1",
        truncation=True, max_length=512,
    )
    _absa_available = True
    logger.info("ABSA model loaded.")
except Exception as _e:
    logger.warning("ABSA unavailable (%s).", _e)

_distilbert_pipeline  = None
_distilbert_available = False
try:
    from transformers import pipeline as hf_pipeline  # noqa: F811
    logger.info("Loading DistilBERT sentiment model ...")
    _distilbert_pipeline = hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True, max_length=512,
    )
    _distilbert_available = True
    logger.info("DistilBERT loaded.")
except Exception as _e:
    logger.warning("DistilBERT unavailable (%s).", _e)

_st_model: Optional[object]          = None
_aspect_embeddings: Optional[object] = None
_aspect_names_ordered: list[str]     = []
_semantic_available                  = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    logger.info("Loading SentenceTransformer (all-MiniLM-L6-v2) ...")
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    _ASPECT_PHRASES = {
        aspect: f"{aspect}: {', '.join(kws)}"
        for aspect, kws in ASPECT_KEYWORDS.items()
    }
    _aspect_names_ordered = list(_ASPECT_PHRASES.keys())
    _aspect_embeddings = _st_model.encode(
        list(_ASPECT_PHRASES.values()),
        convert_to_numpy=True, show_progress_bar=False,
    )
    _semantic_available = True
    logger.info("SentenceTransformer ready.")
except Exception as _e:
    logger.warning("SentenceTransformer unavailable (%s).", _e)

_bertopic_available = False
try:
    from bertopic import BERTopic  # noqa: F401
    _bertopic_available = True
    logger.info("BERTopic available.")
except Exception as _e:
    logger.warning("BERTopic unavailable (%s).", _e)

_ABSA_LABEL_MAP = {
    "positive": ("positive",  1.0),
    "negative": ("negative", -1.0),
    "neutral":  ("neutral",   0.0),
}
_BASE_SEMANTIC_THRESHOLD = 0.48
_KEYWORD_BOOST           = 0.08


# ── Text helpers ───────────────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?;]", text) if len(s.strip()) > 8]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _has_keyword(tokens: list[str], kws: list[str]) -> bool:
    token_set = set(tokens)
    for kw in kws:
        if all(p in token_set for p in kw.split()):
            return True
    return False


# ── IMPROVEMENT 2 + 5: Batch aspect detection ─────────────────────────────────

def _detect_aspects_batch(
    sentences: list[str],
    tokens_list: list[list[str]],
) -> list[list[str]]:
    """
    Detect aspects for ALL sentences in one SentenceTransformer encode call.

    Old approach: encode each sentence individually -> N encode calls
    New approach: encode all sentences at once    -> 1 encode call

    Returns list of matched-aspect lists, one per input sentence.
    """
    if _semantic_available and _st_model is not None:
        try:
            import numpy as np

            # Single encode call for every sentence at once
            sent_embs  = _st_model.encode(
                sentences, convert_to_numpy=True,
                show_progress_bar=False, batch_size=64,
            )
            sent_norms = sent_embs / (
                np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-10
            )
            asp_norms  = _aspect_embeddings / (
                np.linalg.norm(_aspect_embeddings, axis=1, keepdims=True) + 1e-10
            )
            # sim_matrix: (n_sentences, n_aspects)
            sim_matrix = sent_norms @ asp_norms.T

            results = []
            for toks, sims in zip(tokens_list, sim_matrix):
                word_count = len(toks)
                threshold  = _BASE_SEMANTIC_THRESHOLD - min(
                    0.04, max(0.0, (15 - word_count) * 0.003)
                )
                matched = []
                for ai, sim in enumerate(sims):
                    aspect = _aspect_names_ordered[ai]
                    has_kw = _has_keyword(toks, ASPECT_KEYWORDS[aspect])
                    # Require keyword match OR very high semantic similarity
                    # This prevents "quality" matching sound/build/camera all at once
                    if has_kw:
                        boosted = float(sim) + _KEYWORD_BOOST
                        if boosted >= threshold:
                            matched.append(aspect)
                    else:
                        # Pure semantic match needs much higher confidence
                        if float(sim) >= _BASE_SEMANTIC_THRESHOLD + 0.12:
                            matched.append(aspect)
                # Final fallback: keyword-only if nothing matched
                if not matched:
                    matched = [
                        a for a, kws in ASPECT_KEYWORDS.items()
                        if _has_keyword(toks, kws)
                    ]
                results.append(matched)
            return results

        except Exception as exc:
            logger.warning("Batch semantic detection failed (%s); using keywords.", exc)

    return [
        [a for a, kws in ASPECT_KEYWORDS.items() if _has_keyword(toks, kws)]
        for toks in tokens_list
    ]


# ── IMPROVEMENT 1 + 5: Batch sentiment inference ──────────────────────────────

def _lexicon_sentiment(tokens: list[str]) -> tuple[str, float]:
    pos = neg = 0
    for i, t in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in NEGATION_WORDS
        if t in POSITIVE_WORDS:
            neg += 1 if negated else 0
            pos += 0 if negated else 1
        if t in NEGATIVE_WORDS:
            pos += 1 if negated else 0
            neg += 0 if negated else 1
    total = pos + neg
    if total == 0:
        return "neutral", 0.0
    score = (pos - neg) / total
    label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
    return label, round(score, 3)


def _batch_sentiment(
    pairs: list[tuple[str, str, list[str]]]
) -> list[tuple[str, float]]:
    """
    Run sentiment for ALL (sentence, aspect, tokens) pairs in ONE model call.

    Old approach: 1 model call per pair -> 300 calls for 100 reviews x 3 aspects
    New approach: 1 batch call         -> ~10 forward passes total (batch_size=32)

    Tier 1: ABSA batch   (aspect-aware, format: "sentence [SEP] aspect")
    Tier 2: DistilBERT batch (sentence-level fallback)
    Tier 3: Lexicon      (pure Python, always available)
    """
    if not pairs:
        return []

    # Tier 1: ABSA batch
    if _absa_available and _absa_pipeline is not None:
        try:
            inputs  = [f"{sent[:480]} [SEP] {aspect}" for sent, aspect, _ in pairs]
            results = _absa_pipeline(inputs, batch_size=32, truncation=True)
            logger.info("ABSA batch: %d pairs processed.", len(inputs))
            out = []
            for r in results:
                raw   = r["label"].lower().strip()
                conf  = float(r["score"])
                label, base = _ABSA_LABEL_MAP.get(raw, ("neutral", 0.0))
                out.append((label, round(base * conf, 3)))
            return out
        except Exception as exc:
            logger.warning("ABSA batch failed (%s); trying DistilBERT.", exc)

    # Tier 2: DistilBERT batch
    if _distilbert_available and _distilbert_pipeline is not None:
        try:
            inputs  = [sent[:512] for sent, _, _ in pairs]
            results = _distilbert_pipeline(inputs, batch_size=32, truncation=True)
            logger.info("DistilBERT batch: %d pairs processed.", len(inputs))
            out = []
            for r in results:
                raw  = r["label"].lower()
                conf = float(r["score"])
                if raw == "positive":
                    label, score = "positive",  conf
                elif raw == "negative":
                    label, score = "negative", -conf
                else:
                    label, score = "neutral",   0.0
                if abs(score) < 0.55:
                    label, score = "neutral", 0.0
                out.append((label, round(score, 3)))
            return out
        except Exception as exc:
            logger.warning("DistilBERT batch failed (%s); using lexicon.", exc)

    # Tier 3: Lexicon
    return [_lexicon_sentiment(toks) for _, _, toks in pairs]


# ── IMPROVEMENT 3: Richer evidence selection ──────────────────────────────────

def _informativeness(sentence: str) -> float:
    toks = _tokens(sentence)
    n    = len(toks)
    if n == 0:
        return 0.0
    length_score  = min(1.0, n / 20.0) if n <= 20 else max(0.5, 1.0 - (n - 20) / 60.0)
    diversity     = len(set(toks)) / n
    has_sentiment = any(t in POSITIVE_WORDS or t in NEGATIVE_WORDS for t in toks)
    specificity   = 0.15 if has_sentiment else 0.0
    return min(1.0, length_score * 0.4 + diversity * 0.45 + specificity)


def _best_evidence(items: list[tuple[int, float]], evidence: list[str]) -> int:
    """Pick evidence with highest |sentiment_score| x informativeness."""
    best_idx, best_score = items[0][0], -1.0
    for idx, sent_score in items:
        composite = abs(sent_score) * _informativeness(evidence[idx])
        if composite > best_score:
            best_score = composite
            best_idx   = idx
    return best_idx


# ── IMPROVEMENT 4: Helpfulness-weighted confidence ────────────────────────────

def _parse_helpfulness(review: dict) -> float:
    raw = review.get("helpfulness") or review.get("helpful")
    if raw is None:
        return 1.0
    if isinstance(raw, str):
        nums = re.findall(r"\d+", raw)
        if len(nums) < 2:
            return 1.0
        numerator, denominator = int(nums[0]), int(nums[1])
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        numerator, denominator = int(raw[0]), int(raw[1])
    else:
        return 1.0
    if denominator == 0:
        return 1.0
    return round(0.5 + (numerator / denominator) * 1.5, 3)


def _compute_weighted_confidence(reviews: list[dict]) -> dict:
    n = len(reviews)

    # Axis 1: Volume
    if n >= 50:
        volume_level = "high"
    elif n >= 9:
        volume_level = "medium"
    elif n >= 4:
        volume_level = "low"
    else:
        volume_level = "insufficient"

    # Axis 2: Rating agreement (variance)
    ratings = [r["rating"] for r in reviews if r.get("rating") is not None]
    if len(ratings) >= 3:
        mean_r   = sum(ratings) / len(ratings)
        variance = sum((r - mean_r) ** 2 for r in ratings) / len(ratings)
        std_dev  = math.sqrt(variance)
        agreement_level = (
            "high" if std_dev < 0.8 else
            ("medium" if std_dev < 1.5 else "low")
        )
        avg_rating = round(mean_r, 2)
        rating_std = round(std_dev, 2)
    else:
        agreement_level = "medium"
        avg_rating = None
        rating_std = None

    # Axis 3: Helpfulness
    weights         = [_parse_helpfulness(r) for r in reviews]
    avg_helpfulness = sum(weights) / len(weights)
    helpfulness_level = (
        "high" if avg_helpfulness >= 1.4 else
        ("medium" if avg_helpfulness >= 0.9 else "low")
    )

    # Final = weakest axis
    _rank       = {"insufficient": 0, "low": 1, "medium": 2, "high": 3}
    final_level = min(
        [volume_level, agreement_level, helpfulness_level],
        key=lambda lvl: _rank[lvl],
    )

    note_parts = [f"Based on {n} reviews."]
    if rating_std is not None:
        note_parts.append(f"Rating std dev: {rating_std}.")
    note_parts.append(f"Avg helpfulness: {round(avg_helpfulness, 2)}.")

    return {
        "level":             final_level,
        "review_count":      n,
        "avg_rating":        avg_rating,
        "rating_std_dev":    rating_std,
        "avg_helpfulness":   round(avg_helpfulness, 2),
        "volume_level":      volume_level,
        "agreement_level":   agreement_level,
        "helpfulness_level": helpfulness_level,
        "note":              " ".join(note_parts),
    }


# ── Topic discovery ───────────────────────────────────────────────────────────

def _discover_topics(reviews: list[dict]) -> list[dict]:
    if not _bertopic_available:
        return []
    try:
        from bertopic import BERTopic
        docs = []
        for review in reviews:
            body = (review.get("review_text") or "").strip()
            if body and len(body) >= 10:
                docs.extend(s for s in _sentences(body) if len(s) > 15)
        if len(docs) < 10:
            return []
        logger.info("Running BERTopic on %d sentences ...", len(docs))
        topic_model = BERTopic(verbose=False, min_topic_size=max(2, len(docs) // 20))
        topic_model.fit_transform(docs)
        discovered = []
        for _, row in topic_model.get_topic_info().iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                continue
            top_words = [w for w, _ in topic_model.get_topic(tid)[:6]]
            discovered.append({
                "topic_id":             tid,
                "label":                " / ".join(top_words[:3]),
                "representative_words": top_words,
                "count":                int(row["Count"]),
            })
        discovered.sort(key=lambda x: x["count"], reverse=True)
        logger.info("BERTopic discovered %d topics.", len(discovered))
        return discovered
    except Exception as exc:
        logger.error("BERTopic failed: %s", exc, exc_info=True)
        return []


# ── Main analyzer ─────────────────────────────────────────────────────────────

class ReviewAnalyzer:
    MIN_REVIEWS = 4   # below 4 -> insufficient

    def __init__(self, reviews: list[dict]):
        self.reviews = reviews

    def analyze(self) -> dict:
        if len(self.reviews) < self.MIN_REVIEWS:
            return self._not_enough_data()

        aspect_data        = self._extract_aspects()
        top_aspects        = self._rank_aspects(aspect_data)
        pros, cons         = self._build_pros_cons(aspect_data)
        confidence         = _compute_weighted_confidence(self.reviews)
        summary            = self._build_summary(top_aspects, pros, cons)
        discovered_aspects = _discover_topics(self.reviews)

        return {
            "top_aspects":        top_aspects,
            "pros":               pros,
            "cons":               cons,
            "summary":            summary,
            "confidence":         confidence,
            "review_count":       len(self.reviews),
            "discovered_aspects": discovered_aspects,
        }

    # ── Two-pass batched extraction ───────────────────────────────────────────
    #
    # Pass 1: collect ALL sentences from all reviews
    # Pass 2: one encode call detects aspects for all sentences
    # Pass 3: build flat (sentence, aspect) pair list
    # Pass 4: one model call scores sentiment for all pairs
    # Pass 5: assemble results into aspect_data
    #
    # Replaces old per-sentence-per-aspect loop that called the model
    # hundreds of times for a single product.

    def _extract_aspects(self) -> dict:

        # Pass 1: collect sentences
        all_sentences : list[str]       = []
        all_tokens    : list[list[str]] = []
        all_ratings   : list           = []
        all_hw        : list[float]    = []

        for review in self.reviews:
            body   = (review.get("review_text") or "").strip()
            rating = review.get("rating")
            hw     = _parse_helpfulness(review)

            if not body or len(body) < 10:
                continue

            for sent in _sentences(body):
                toks = _tokens(sent)
                all_sentences.append(sent)
                all_tokens.append(toks)
                all_ratings.append(rating)
                all_hw.append(hw)

        if not all_sentences:
            return {}

        # Pass 2: batch aspect detection (one encode call)
        aspects_per_sentence = _detect_aspects_batch(all_sentences, all_tokens)

        # Pass 3: build flat pairs list
        pairs     : list[tuple[str, str, list[str]]] = []
        pair_meta : list[tuple[int, str]]            = []

        for si, matched in enumerate(aspects_per_sentence):
            for aspect in matched:
                pairs.append((all_sentences[si], aspect, all_tokens[si]))
                pair_meta.append((si, aspect))

        if not pairs:
            return {}

        # Pass 4: batch sentiment (one model call)
        sentiments = _batch_sentiment(pairs)

        # Pass 5: assemble aspect_data
        aspect_data: dict[str, dict] = defaultdict(
            lambda: {"sentiments": [], "evidence": [], "helpfulness": []}
        )

        for (si, aspect), (label, score) in zip(pair_meta, sentiments):
            rating = all_ratings[si]
            hw     = all_hw[si]

            if rating:
                bias  = (float(rating) - 3.0) * 0.15
                score = max(-1.0, min(1.0, score + bias))
                label = (
                    "positive" if score > 0.1
                    else ("negative" if score < -0.1 else "neutral")
                )

            aspect_data[aspect]["sentiments"].append((label, score))
            aspect_data[aspect]["evidence"].append(all_sentences[si])
            aspect_data[aspect]["helpfulness"].append(hw)

        return aspect_data

    # ── Ranking ───────────────────────────────────────────────────────────────

    def _rank_aspects(self, aspect_data: dict) -> list[dict]:
        ranked = []
        for aspect, data in aspect_data.items():
            sents = data["sentiments"]
            if not sents:
                continue
            hw      = data["helpfulness"]
            total_w = sum(hw)
            avg     = (
                sum(s * w for (_, s), w in zip(sents, hw)) / total_w
                if total_w > 0
                else sum(s for _, s in sents) / len(sents)
            )
            pos   = sum(1 for l, _ in sents if l == "positive")
            neg   = sum(1 for l, _ in sents if l == "negative")
            label = "positive" if avg > 0.05 else ("negative" if avg < -0.05 else "mixed")
            ranked.append({
                "aspect":            aspect,
                "sentiment":         label,
                "score":             round(avg, 3),
                "mention_count":     len(sents),
                "positive_mentions": pos,
                "negative_mentions": neg,
            })
        ranked.sort(key=lambda x: x["mention_count"], reverse=True)
        return ranked[:8]

    # ── Pros / cons ───────────────────────────────────────────────────────────

    def _build_pros_cons(self, aspect_data: dict) -> tuple[list, list]:
        pros, cons = [], []
        for aspect, data in aspect_data.items():
            sents    = data["sentiments"]
            evidence = data["evidence"]
            if not sents:
                continue
            pos_items = [(i, s) for i, (l, s) in enumerate(sents) if l == "positive"]
            neg_items = [(i, s) for i, (l, s) in enumerate(sents) if l == "negative"]

            if pos_items and len(pos_items) >= len(neg_items):
                best_idx = _best_evidence(pos_items, evidence)
                pros.append({
                    "aspect":      aspect,
                    "description": f"Positively mentioned in {len(pos_items)} review(s).",
                    "evidence":    evidence[best_idx][:250],
                })
            if neg_items and len(neg_items) >= max(1, len(pos_items)):
                worst_idx = _best_evidence(neg_items, evidence)
                cons.append({
                    "aspect":      aspect,
                    "description": f"Negatively mentioned in {len(neg_items)} review(s).",
                    "evidence":    evidence[worst_idx][:250],
                })
        return pros[:6], cons[:6]

    # ── Summary ───────────────────────────────────────────────────────────────

    def _build_summary(self, top_aspects, pros, cons) -> str:
        pos = [a["aspect"] for a in top_aspects if a["sentiment"] == "positive"]
        neg = [a["aspect"] for a in top_aspects if a["sentiment"] == "negative"]
        parts = []
        if pos:
            parts.append(f"Customers appreciate {', '.join(pos[:3])}.")
        if neg:
            parts.append(f"Common concerns include {', '.join(neg[:2])}.")
        if not parts:
            parts.append("Mixed customer sentiment with no dominant theme.")
        return " ".join(parts)

    # ── Not enough data ───────────────────────────────────────────────────────

    @staticmethod
    def _not_enough_data() -> dict:
        return {
            "top_aspects":        [],
            "pros":               [],
            "cons":               [],
            "summary":            "Not enough reviews to generate reliable insights.",
            "confidence": {
                "level":             "insufficient",
                "review_count":      0,
                "avg_rating":        None,
                "rating_std_dev":    None,
                "avg_helpfulness":   None,
                "volume_level":      "insufficient",
                "agreement_level":   "insufficient",
                "helpfulness_level": "insufficient",
                "note":              "Minimum 4 reviews required.",
            },
            "review_count":       0,
            "discovered_aspects": [],
        }