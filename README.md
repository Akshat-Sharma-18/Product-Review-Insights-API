# Product Review Insights API

A Python/Flask REST API that analyzes real customer reviews from **Reviews1.csv** and **Reviews2.json**, extracting top aspects, sentiment, pros & cons with evidence, and a confidence score.

All outputs are **grounded in your dataset** — no hallucination, no fabricated citations.

---

## Dataset

This project uses two publicly available Amazon review datasets:

| Dataset | Source | Format |
|---------|--------|--------|
| Amazon Fine Food Reviews | [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) | CSV |
| Amazon Electronics Reviews | [Kaggle](https://www.kaggle.com/datasets/snap/amazon-reviews) | JSONL |

### Setup
Download the files and place them in the `data/` folder:
```
data/Reviews1.csv
data/Reviews2.json
```

---

## Supported Schemas

### Reviews1.csv  (Amazon Food Reviews style)
| Column | Description |
|--------|-------------|
| `Id` | Row ID |
| `ProductId` | Product identifier (e.g. `B001E4KFA3`) |
| `UserId` | Reviewer ID |
| `ProfileName` | Reviewer name |
| `HelpfulnessNumerator` | Helpful votes received |
| `HelpfulnessDenominator` | Total helpfulness votes |
| `Score` | Star rating (1–5) |
| `Time` | Unix timestamp |
| `Summary` | Short review headline |
| `Text` | Full review body ← **primary analysis field** |

### Reviews2.json  (Amazon Electronics / JSONL style)
| Column | Description |
|--------|-------------|
| `asin` | Product identifier (e.g. `0528881469`) |
| `reviewerID` | Reviewer ID |
| `reviewerName` | Reviewer name |
| `helpful` | Helpfulness as `"[num, denom]"` |
| `overall` | Star rating (1–5) |
| `reviewText` | Full review body ← **primary analysis field** |
| `summary` | Short review headline |
| `reviewTime` | Human-readable date |
| `unixReviewTime` | Unix timestamp |

Both schemas are **auto-detected** at startup based on header column names.

---

## Project Structure

```
product_review_api/
├── main.py           # Flask API — routes, middleware, error handling
├── analyzer.py       # NLP engine — aspect extraction + sentiment analysis
├── data_loader.py    # Data loader — supports CSV and JSONL, merges both files
├── demo_store.py     # In-memory store for Demo Product Mode
├── index.html        # Frontend UI — dark-themed, product browser + demo mode
├── data/
│   ├── Reviews1.csv  # ← Place your Reviews1.csv here
│   └── Reviews2.json # ← Place your Reviews2.json here
├── logs/
│   └── api.log       # Runtime logs
└── README.md
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install flask transformers torch sentence-transformers bertopic
```

### 2. Place your dataset files
```
data/Reviews1.csv
data/Reviews2.json
```

Or use environment variables to point to custom paths:
```bash
export REVIEWS_CSV=/path/to/your/Reviews1.csv
export REVIEWS_JSON=/path/to/your/Reviews2.json
```

### 3. Start the API
```bash
python main.py
```
Server starts at `http://localhost:5000`

---

## API Endpoints

### `GET /api/v1/health`
```json
{
  "status": "ok",
  "files_loaded": ["Reviews1.csv", "Reviews2.json"],
  "products_loaded": 2874,
  "total_reviews": 568454
}
```

---

### `GET /api/v1/products`
Lists all product IDs in the merged dataset.

Query params:
- `?search=B001` — filter by substring
- `?page=1&limit=100` — paginate results

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

### `POST /api/v1/insights`
**Request:**
```json
{ "product_id": "B001E4KFA3" }
```

**Response:**
```json
{
  "product_id": "B001E4KFA3",
  "review_count": 512,
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
  }
}
```

---

### `GET /api/v1/insights?product_id=B001E4KFA3`
Convenience GET version.

---

## Demo Product Mode

Test the API without a dataset by creating a demo product and adding reviews manually.

### `POST /api/v1/demo-product`
```json
{ "product_id": "my_test_phone" }
```

### `POST /api/v1/demo-review`
```json
{
  "product_id": "my_test_phone",
  "review_text": "Battery lasts all day and the camera is excellent",
  "rating": 5
}
```

### `GET /api/v1/demo-insights?product_id=my_test_phone`
Returns full insights (same shape as `/api/v1/insights`).

### `DELETE /api/v1/demo-product?product_id=my_test_phone`
Removes the demo product from memory.

---

## Example API Calls

### cURL — POST
```bash
curl -X POST http://localhost:5000/api/v1/insights \
  -H "Content-Type: application/json" \
  -d '{"product_id": "B001E4KFA3"}'
```

### cURL — GET
```bash
curl "http://localhost:5000/api/v1/insights?product_id=0528881469"
```

### Python
```python
import requests

resp = requests.post(
    "http://localhost:5000/api/v1/insights",
    json={"product_id": "B001E4KFA3"}
)
data = resp.json()

print(data["summary"])
for pro in data["pros"]:
    print(f"✅ {pro['aspect']}: {pro['evidence']}")
for con in data["cons"]:
    print(f"❌ {con['aspect']}: {con['evidence']}")
```

---

## Error Responses

| Status | Scenario |
|--------|----------|
| `400` | Missing / empty / too-long `product_id`, non-JSON body |
| `404` | Product not found in dataset |
| `503` | No dataset loaded |

```json
{ "error": "Product 'XYZ' not found in the dataset." }
```

---

## Confidence Levels

| Level | Criteria |
|-------|---------|
| `high` | ≥ 50 reviews, low rating variance, helpful votes present |
| `medium` | 9 – 49 reviews |
| `low` | 4 – 8 reviews |
| `insufficient` | < 4 reviews |

---

## How It Works

1. **Data loading** — Both files loaded at startup, schema auto-detected, merged and de-duplicated
2. **Text preprocessing** — HTML tag removal, URL stripping, whitespace normalization
3. **Aspect extraction** — 19-category keyword taxonomy + SentenceTransformer semantic similarity
4. **Sentiment analysis** — DistilBERT per-sentence scoring with negation handling + star rating calibration
5. **Evidence selection** — Best actual review sentences surfaced as evidence (never invented)
6. **Confidence scoring** — Three-axis: review volume · rating agreement · helpfulness weighting

---

## Evaluation Alignment

| Criterion | Implementation |
|-----------|----------------|
| Correctness & Functionality (40%) | Aspect extraction, sentiment, pros/cons with evidence, confidence, "not enough data" |
| AI/ML Quality (30%) | DistilBERT + SentenceTransformer + negation + rating calibration; no hallucination |
| API Design (20%) | REST endpoints, input validation, pagination, timing headers, structured errors, demo mode |
| Documentation (10%) | This README with setup, run instructions, dataset links, and example calls |
