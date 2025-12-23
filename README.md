# Starship Coffee — Take-Home (RAG + Tool Calling + Receipt OCR)

This repo contains my implementation of the **3 Streamlit apps** required for the Starship Coffee take-home assignment:

- **Task 1 (RAG + citations + injection handling)** → `part1_rag/app_rag.py`
- **Task 2 (Function calling over CSVs)** → `part2_funcs/app_funcs.py`
- **Task 3 (Receipt OCR using a vision model)** → `part3_receipts/app_receipts.py`

> ⚠️ **Note about cost/quota:** During initial runs, using OpenAI Embeddings triggered a **429 / insufficient_quota**.  
> For that reason, in **Task 1** I kept **generation via OpenAI** but moved **embeddings to a local TF-IDF** approach (no cost), which still satisfies the “embed + retrieve + answer with citations” pipeline for the demo.

---

## ✅ Quickstart
```bash
 1) Install dependencies

pip install -r requirements.txt
2) Create .env
Copy the template:

bash
Copy code
cp env.example .env
Fill in:

env
Copy code
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
3) Run apps
bash
Copy code
 Task 1
streamlit run part1_rag/app_rag.py

 Task 2
streamlit run part2_funcs/app_funcs.py

 Task 3
streamlit run part3_receipts/app_receipts.py
```
---
## Task 1 — Simple RAG (Citations + Injection Handling)
What this app does
Reads markdown docs from: part1_rag/docs/

Chunks docs into overlapping windows (character-based chunking).

Builds embeddings locally (TF-IDF) for all chunks (no API cost).

Retrieves top-k chunks by cosine similarity.

Uses OpenAI only for answer generation:

Answer is constrained to ≤ 100 words

The prompt instructs: use only retrieved text

Returns citations as a list of filenames (doc_id).

Storage backends (implemented)
✅ JSON store: part1_rag/.store/store.json

✅ SQLite store: part1_rag/.store/store.sqlite

Prompt-Injection handling
If the user requests anything like:

revealing secret files

printing contents of secrets/

dumping hidden info
…then the app refuses with one sentence.

Key design note (important)
Because TF-IDF uses a fitted vocabulary, when chunks are loaded from disk (JSON/SQLite) the vectorizer must be re-initialized from the loaded chunk texts.
This is handled by ensure_vectorizer() after loading.

Screenshot (Task 1 output)
Replace the placeholder below with your screenshot.

---
## Task 2 — Function Calling over Customer/Order CSVs
What this app does
Loads two CSVs from: part2_funcs/data/

customers.csv

orders.csv

Exposes three tools/functions to the model:

get_order(order_id) → returns status, total, masked_email

refund_order(order_id, amount) → returns {ok, reason?}

Allowed only if status in {settled, prepping}

Allowed only if amount <= total

spend_in_period(customer_id, start, end) → returns total_spend

Uses OpenAI tool calling to:

choose which tool(s) to call

call them with correct args

assemble a final answer

UI behavior
Left side: free text + preset buttons

Right side: final answer + tool call table (tool, args, result)

Emails are always masked (e.g. l***@domain.com)

Screenshot (Task 2 output)
Replace the placeholder below with your screenshot.
---

## Task 3 — Receipt OCR (Vision → Structured JSON)
What this app does
Accepts a single receipt image upload (PNG/JPG)

Calls a hosted vision model to extract:

items: list of {name, qty, unit_price, line_total}

total

Displays:

JSON output

Table view of items

Logs each prediction to a local .jsonl file:

part3_receipts/predictions.jsonl

Edge case handled
If a receipt contains both a crossed-out total and a current total, the prompt instructs the model to return the current total.

Screenshot (Task 3 output)
Replace the placeholder below with your screenshot.
---

##  Repo Notes
What is NOT committed
.env (contains API key)

.venv/ virtual environment

local stores / cached indexes (.store/, sqlite/db files)

local logs (*.jsonl)
---

## How I would switch Task 1 back to OpenAI embeddings (if quota is available)

If embeddings quota/billing is enabled:

replace TF-IDF embedding functions with:

text-embedding-3-small or similar

store vectors in JSON/SQLite as currently implemented

retrieval remains the same

