# Product Review Insights API

A production-grade Flask REST API that turns raw customer reviews into structured intelligence — extracting aspects, sentiment, pros, cons, and emergent topics — using a **multi-model hybrid NLP pipeline** built entirely from specialised, efficient transformers.

All outputs are **grounded in your actual review data**. Nothing is invented, summarised by a language model, or hallucinated.

---

## Table of Contents

- [Why a Hybrid NLP Pipeline — Not an LLM](#why-a-hybrid-nlp-pipeline--not-an-llm)
- [How the Hybrid Pipeline Works](#how-the-hybrid-pipeline-works)
- [The Four Models and Their Roles](#the-four-models-and-their-roles)
- [Full Pipeline Flow](#full-pipeline-flow)
- [Performance Architecture](#performance-architecture)
- [Project Structure](#project-structure)
- [Setup & Run](#setup--run)
- [API Endpoints](#api-endpoints)
- [Demo Product Mode](#demo-product-mode)
- [Cache Admin Endpoints](#cache-admin-endpoints)
- [API Response Reference](#api-response-reference)
- [Error Responses](#error-responses)
- [Confidence Levels](#confidence-levels)
- [Data Sources & Schemas](#data-sources--schemas)
- [Environment Variables](#environment-variables)

---

## Why a Hybrid NLP Pipeline — Not an LLM

When designing the sentiment and aspect analysis engine, the most obvious modern approach would have been to send reviews to a large language model like GPT-4, Claude, or Llama and ask it to extract pros, cons, and sentiments in a prompt. We deliberately chose not to do this, for the following reasons.

**LLMs hallucinate.** When asked to summarise or analyse reviews, LLMs frequently generate plausible-sounding sentences that do not appear anywhere in the source data. For a review intelligence product, this is a fundamental correctness failure. Every piece of evidence surfaced by this system is an exact sentence from an actual review — nothing is generated.

**LLMs are expensive at scale.** Running GPT-4 over 500 reviews per product, across thousands of products, produces API costs that compound rapidly. The specialised models used here run locally, with no per-call cost after the initial download.

**LLMs are slow for batch workloads.** A single LLM call for a product with 200 reviews would need to handle the full context window carefully, or be split into many sequential calls. Specialised transformer models run in parallel batches and complete analysis orders of magnitude faster.

**LLMs are black boxes for structured extraction.** Getting a consistent, schema-stable JSON output from an LLM requires careful prompt engineering, output parsing, and retry logic. Structured models produce deterministic, typed outputs directly.

**LLMs have no aspect-level precision.** A sentence like *"Battery is great but the screen is terrible"* would be summarised by an LLM as "mixed" without necessarily attributing the correct sentiment to each specific aspect. The ABSA model in this pipeline evaluates each aspect independently within the sentence, producing correct per-aspect polarity.

**The right tool for each job.** LLMs are generalists. Each model in this pipeline was trained specifically for its task: semantic similarity, sentence classification, aspect-based sentiment, and topic clustering. Combining them produces higher accuracy than asking a generalist to do everything.

---

## How the Hybrid Pipeline Works

The NLP engine is built as a layered pipeline where each layer handles one specific problem. No single model carries all the responsibility.

```
Raw review text
      |
      v
Sentence splitting  (regex, once per review, results reused everywhere)
      |
      v
Semantic Aspect Detection  (SentenceTransformer all-MiniLM-L6-v2)
      |  cosine similarity against 19 canonical aspect embeddings
      |  fallback: keyword taxonomy matching
      v
Aspect-Aware Sentiment  (DeBERTa ABSA -- per aspect, per sentence)
      |  format: "{sentence} [SEP] {aspect}"
      |  fallback tier 2: DistilBERT sentence-level (batched)
      |  fallback tier 3: lexicon + negation rules
      v
Star-rating calibration  (bias = (rating - 3.0) * 0.15)
      v
Evidence extraction  (best/worst real sentence per aspect)
      v
Pros / Cons generation  (evidence-backed, nothing invented)
      v
Confidence scoring  (based on review volume)
      v
Topic discovery  (BERTopic -- only when >= 30 reviews)
      v
Structured API response
```

Each model loads once at startup and is reused across all requests. Sentence embeddings are cached across calls so repeated text is never re-encoded.

---

## The Four Models and Their Roles

### 1. `all-MiniLM-L6-v2` — Semantic Aspect Detection

**Provider:** SentenceTransformers
**Task:** Determine which product aspects are being discussed in a sentence.

This model converts sentences into dense vector embeddings in a shared semantic space. We pre-compute embeddings for 19 canonical aspect phrases (e.g. *"battery life battery charge charging last power"*) at startup. For each review sentence, we encode it and compute cosine similarity against all 19 aspect embeddings. Any aspect with similarity above the 0.50 threshold is considered mentioned.

This approach handles natural language variation cleanly. A sentence like *"the charge runs out so fast"* correctly matches `battery life` even though it contains none of the obvious trigger words. Pure keyword matching would miss it entirely.

If the model is unavailable, the system falls back to a keyword taxonomy (19 aspect categories, ~150 total trigger terms) with multi-word phrase support.

---

### 2. `yangheng/deberta-v3-base-absa-v1.1` — Aspect-Based Sentiment Analysis (ABSA)

**Provider:** HuggingFace
**Task:** Classify the sentiment toward a *specific aspect* within a sentence.

This is the core accuracy improvement over conventional sentiment analysis. Standard sentiment models evaluate sentiment for an entire sentence and return one label. That label is then incorrectly assigned to every aspect found in that sentence.

The ABSA model accepts the sentence and the target aspect together, formatted as:

```
"{sentence} [SEP] {aspect}"
```

This lets the model attend to the relevant portion of the sentence for each aspect individually. For a sentence like:

```
Camera is amazing but battery drains quickly
```

The system makes two separate model calls:

```
"Camera is amazing but battery drains quickly [SEP] camera"   -> positive (0.97)
"Camera is amazing but battery drains quickly [SEP] battery"  -> negative (0.94)
```

The signed score is `confidence * direction` (+1 for positive, -1 for negative) so that uncertain predictions near 0.5 confidence contribute proportionally less to the aspect average.

If this model is unavailable, the pipeline falls back to the sentence-level DistilBERT result for that aspect. If that also fails, it uses lexicon rules.

---

### 3. `distilbert-base-uncased-finetuned-sst-2-english` — Sentence-Level Sentiment Fallback

**Provider:** HuggingFace
**Task:** Sentence-level positive/negative classification, used as ABSA fallback and for pre-computation.

This model runs in a single batched call over all unique sentences in the review set before the per-aspect loop begins. The results are stored in a dictionary keyed by sentence text. When the ABSA model fails for a given aspect, the pre-computed sentence-level result is returned immediately at no additional inference cost — there is no second model call.

Predictions with confidence below 0.55 are mapped to neutral to avoid overconfident labelling of ambiguous text.

If batch inference fails entirely, the system falls back to per-sentence lexicon scoring.

---

### 4. `BERTopic` — Emergent Topic Discovery

**Provider:** MaartenGr/BERTopic
**Task:** Discover topic clusters in review text that fall outside the predefined 19-aspect taxonomy.

BERTopic applies a clustering pipeline (UMAP dimensionality reduction + HDBSCAN clustering + c-TF-IDF topic representation) to all review sentences, surfacing groups of semantically related sentences without any predefined labels. This catches emerging themes — a product flaw not in the taxonomy, an unexpected use case, a recurring packaging complaint — that keyword-based or embedding-based matching would miss.

BERTopic is computationally expensive. It is skipped entirely when the review set has fewer than 30 reviews (not worth the latency) and when fewer than 10 sentences can be extracted. Results appear in the `discovered_aspects` field of the API response.

If BERTopic is unavailable, `discovered_aspects` is returned as an empty array and the rest of the pipeline is unaffected.

---

## Full Pipeline Flow

```
Startup
  load all-MiniLM-L6-v2
  pre-compute 19 aspect embeddings
  load DeBERTa ABSA model
  load DistilBERT sentiment model
  check BERTopic availability
  load CSV / JSONL datasets into REVIEWS_DB

Request: GET /api/v1/insights?product_id=XYZ
  |
  +-- cache hit?  -> return cached result (~20ms)
  |
  +-- cache miss
        |
        fetch reviews from REVIEWS_DB
        |
        ReviewAnalyzer.analyze()
          |
          pre-split all reviews into sentences (once)
          batch-encode new sentence embeddings -> EMBEDDING_CACHE
          batch DistilBERT over all unique sentences
          |
          for each sentence:
            detect aspects  (semantic or keyword)
            for each aspect:
              call ABSA model  -> (label, score)
              fallback: use pre-computed DistilBERT result
              fallback: use lexicon
              apply star-rating calibration
              append to aspect_data[aspect]
          |
          rank aspects by mention count
          select best evidence sentence per aspect
          build pros / cons
          compute confidence
          build summary
          run BERTopic (if >= 30 reviews)
        |
        store in INSIGHT_CACHE (TTL: 1 hour)
        return JSON response
```

---

## Performance Architecture

The system is designed to hit sub-100ms response times on repeated requests and complete a first analysis in 2-4 seconds rather than 10+.

**TTL Insight Cache.** Results are cached in a TTLCache (default 1 hour, max 500 entries) keyed by `product_id`. Cache hits return the full pre-computed result with no model inference. Uses `cachetools.TTLCache` when installed, otherwise a built-in ordered-dict implementation.

**Sentence Embedding Cache.** The `EMBEDDING_CACHE` dict stores sentence text to embedding vector at the process level. On subsequent calls for the same product (or across products sharing review phrasing), embeddings are never recomputed.

**Batch DistilBERT Inference.** All unique sentences across the entire review set are passed to DistilBERT in a single batched call (`batch_size=64`). This is 3-5x faster than per-sentence calls because the model amortises tokenisation and forward-pass overhead across the full batch.

**BERTopic Threshold.** Topic clustering is skipped when `len(reviews) < 30`, eliminating the most expensive step for small and demo review sets.

**Sentence Pre-Splitting.** Each review is split into sentences exactly once. The resulting `(sentence, rating)` list is reused by the embedding step, the batch inference step, and the aspect extraction loop — no repeated regex work.

**Async Mode.** Setting `ANALYZE_ASYNC=true` makes the API return a `202 Analysis Pending` response immediately and run the pipeline in a background daemon thread. The result is written to the cache when ready. Subsequent requests for the same product will hit the cache.

**Per-request timing headers.** Every response includes `X-Process-Time-Ms` so you can observe cache hit vs miss latency from the client.

Expected performance:

| Scenario | Latency |
|----------|---------|
| Cache hit (any size) | ~20-50 ms |
| First analysis, 50 reviews, all models | ~2-4 s |
| First analysis, 500 reviews, all models | ~8-15 s |
| First analysis, async mode | <50 ms (202 response) |

---

## Project Structure

```
product_review_api/
├── main.py           # Flask API -- routes, cache, middleware, error handling
├── analyzer.py       # Hybrid NLP engine -- all four models, fallback chain
├── data_loader.py    # Dataset loader -- CSV + JSONL, schema detection, deduplication
├── demo_store.py     # In-memory demo product and review storage
├── data/
│   ├── Reviews1.csv  # Amazon Food Reviews (CSV format)
│   └── Reviews2.json # Amazon Electronics (JSONL format)
├── logs/
│   └── api.log       # Runtime logs (UTF-8 encoded)
└── README.md
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install flask
pip install transformers torch
pip install sentence-transformers
pip install bertopic                 # optional -- topic discovery
pip install cachetools               # optional -- thread-safe TTL cache
```

### 2. Place your data files

```
data/Reviews1.csv
data/Reviews2.json
```

Or set custom paths via environment variables (see [Environment Variables](#environment-variables)).

### 3. Start the API

```bash
python main.py
```

Server starts at `http://localhost:5000`. Models are downloaded from HuggingFace on first run and cached locally by the transformers library.

---

## API Endpoints

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "files_loaded": ["Reviews1.csv", "Reviews2.json"],
  "products_loaded": 2874,
  "total_reviews": 568454,
  "cache_entries": 12,
  "async_mode": false
}
```

---

### `GET /api/v1/products`

Lists all product IDs in the merged dataset.

Query params:
- `?search=B001` -- filter by substring
- `?page=1&limit=100` -- paginate results

```json
{
  "products": ["0528881469", "B000FA64PK", "B001E4KFA3"],
  "total": 2874,
  "page": 1,
  "limit": 100,
  "pages": 29
}
```

---

### `GET /api/v1/insights?product_id=B001E4KFA3`
### `POST /api/v1/insights` -- body: `{"product_id": "B001E4KFA3"}`

Returns full NLP analysis. First call runs the pipeline and caches the result. Subsequent calls return the cache.

```json
{
  "product_id": "B001E4KFA3",
  "review_count": 512,
  "cached": false,
  "top_aspects": [
    {
      "aspect": "food quality",
      "sentiment": "positive",
      "score": 0.72,
      "mention_count": 284,
      "positive_mentions": 251,
      "negative_mentions": 33
    }
  ],
  "pros": [
    {
      "aspect": "food quality",
      "description": "Positively mentioned in 251 review(s).",
      "evidence": "I have bought several of the Vitality canned dog food products and have found them all to be of good quality"
    }
  ],
  "cons": [
    {
      "aspect": "price/value",
      "description": "Negatively mentioned in 47 review(s).",
      "evidence": "For the price I expected much better quality"
    }
  ],
  "summary": "Customers appreciate food quality, taste/flavor, shipping. Common concerns include price/value.",
  "confidence": {
    "level": "high",
    "review_count": 512,
    "note": "Based on 512 reviews. More reviews improve accuracy."
  },
  "discovered_aspects": [
    {
      "topic_id": 0,
      "label": "smell / odor / scent",
      "representative_words": ["smell", "odor", "scent", "strong", "awful", "chemical"],
      "count": 47
    }
  ]
}
```

---

## Demo Product Mode

Allows you to test the full NLP pipeline with your own review text without touching the loaded dataset.

### `POST /api/v1/demo-product` -- create a product

```json
{ "product_id": "demo_phone_1" }
```

### `POST /api/v1/demo-review` -- add a review

```json
{
  "product_id": "demo_phone_1",
  "review_text": "Camera is amazing but the battery drains way too fast",
  "summary": "Good camera, terrible battery",
  "rating": 3
}
```

### `GET /api/v1/demo-insights?product_id=demo_phone_1`

Runs the full pipeline on the in-memory reviews. Response shape is identical to `/api/v1/insights`. Demo results are not cached (reviews may change between calls).

### `DELETE /api/v1/demo-product?product_id=demo_phone_1`

Removes the demo product and all its reviews from memory.

### `GET /api/v1/demo-products`

Lists all demo products currently in memory.

**Limits:** max 100 reviews per demo product, max 50 demo products total. All data is lost on server restart.

---

## Cache Admin Endpoints

### `GET /api/v1/cache-stats`

```json
{
  "entries": 12,
  "maxsize": 500,
  "ttl_sec": 3600,
  "async_mode": false
}
```

### `DELETE /api/v1/cache?product_id=B001E4KFA3`

Evicts a single product from the insight cache, forcing a fresh analysis on the next request.

### `DELETE /api/v1/cache`

Flushes the entire insight cache.

---

## API Response Reference

### `top_aspects`

Up to 8 most-mentioned aspects, sorted by mention count.

| Field | Type | Description |
|-------|------|-------------|
| `aspect` | string | Canonical aspect name (e.g. `"battery life"`) |
| `sentiment` | string | `positive`, `negative`, or `mixed` |
| `score` | float | Average signed sentiment score across all mentions |
| `mention_count` | int | Total sentences mentioning this aspect |
| `positive_mentions` | int | Count of positive-sentiment mentions |
| `negative_mentions` | int | Count of negative-sentiment mentions |

### `pros` / `cons`

Up to 6 each. Only generated when enough evidence exists.

| Field | Type | Description |
|-------|------|-------------|
| `aspect` | string | Aspect name |
| `description` | string | e.g. `"Positively mentioned in 34 review(s)."` |
| `evidence` | string | Actual review sentence (max 250 chars) -- never generated |

### `discovered_aspects`

Topics found by BERTopic outside the predefined taxonomy. Empty array when BERTopic is unavailable or skipped.

| Field | Type | Description |
|-------|------|-------------|
| `topic_id` | int | BERTopic cluster ID |
| `label` | string | Top-3 representative words joined by ` / ` |
| `representative_words` | list | Top-6 terms for this topic |
| `count` | int | Number of sentences in this cluster |

---

## Error Responses

| Status | Scenario |
|--------|----------|
| `400` | Missing / empty / invalid `product_id`, non-JSON body |
| `404` | Product not found in dataset |
| `202` | Async mode: analysis queued, retry shortly |
| `503` | No dataset loaded |

```json
{ "error": "Product 'XYZ' not found in dataset." }
```

---

## Confidence Levels

| Level | Condition |
|-------|-----------|
| `high` | >= 50 reviews |
| `medium` | 3 - 49 reviews |
| `low` | < 3 reviews |
| `insufficient` | 0 -- not enough data to analyse |

---

## Data Sources & Schemas

### Reviews1.csv -- Amazon Food Reviews (CSV)

| Column | Description |
|--------|-------------|
| `ProductId` | Product identifier |
| `Score` | Star rating (1-5) |
| `Text` | Full review body -- primary analysis field |
| `Summary` | Short headline |
| `UserId` | Reviewer ID |
| `Time` | Unix timestamp |

### Reviews2.json -- Amazon Electronics (JSONL)

| Field | Description |
|-------|-------------|
| `asin` | Product identifier |
| `overall` | Star rating (1-5) |
| `reviewText` | Full review body -- primary analysis field |
| `summary` | Short headline |
| `reviewerID` | Reviewer ID |
| `reviewTime` | Human-readable date |

Both formats are auto-detected at startup. Records are merged and de-duplicated on `(reviewer_id, first 80 chars of review_text)`. Reviews shorter than 15 characters are discarded.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Flask port |
| `REVIEWS_CSV` | `data/Reviews1.csv` | Path to CSV dataset |
| `REVIEWS_JSON` | `data/Reviews2.json` | Path to JSONL dataset |
| `INSIGHT_CACHE_TTL` | `3600` | Insight cache TTL in seconds |
| `ANALYZE_ASYNC` | `false` | Set `true` to return 202 and analyse in background |

---

## Example Usage

### cURL

```bash
# Check health
curl http://localhost:5000/api/v1/health

# Analyse a product
curl "http://localhost:5000/api/v1/insights?product_id=B001E4KFA3"

# POST form
curl -X POST http://localhost:5000/api/v1/insights \
  -H "Content-Type: application/json" \
  -d '{"product_id": "B001E4KFA3"}'

# Check cache stats
curl http://localhost:5000/api/v1/cache-stats

# Force re-analysis
curl -X DELETE "http://localhost:5000/api/v1/cache?product_id=B001E4KFA3"
```

### Python

```python
import requests

BASE = "http://localhost:5000"

# Create a demo product and add reviews
requests.post(f"{BASE}/api/v1/demo-product",
              json={"product_id": "my_test"})

reviews = [
    ("Battery is great but the screen is absolutely terrible", 2),
    ("Camera takes stunning photos, very happy with the quality", 5),
    ("Fast delivery but the packaging was damaged on arrival", 3),
]
for text, rating in reviews:
    requests.post(f"{BASE}/api/v1/demo-review",
                  json={"product_id": "my_test", "review_text": text, "rating": rating})

# Analyse
r = requests.get(f"{BASE}/api/v1/demo-insights", params={"product_id": "my_test"})
data = r.json()

print(data["summary"])
for pro in data["pros"]:
    print(f"  + {pro['aspect']}: {pro['evidence']}")
for con in data["cons"]:
    print(f"  - {con['aspect']}: {con['evidence']}")
```
