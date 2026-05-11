# Support_Ticket_Classification

Support ticket classifier using Gemini embeddings with a FAISS index built over
representative embeddings per category, plus cross-encoder reranking.

## Quick Setup (Local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (ignored by git) and add your key:

```bash
GEMINI_API_KEY=your_key_here
```

If `faiss-cpu` fails to install on Windows, try:

```bash
conda install -c pytorch faiss-cpu
```

## Build the Representative Index

```bash
python build_representative_index.py --samples-per-category 10 --seed 42
```

This writes:

- `faiss_category.index`
- `faiss_category.meta.json`

Supported dataset column names:

- ticket text: `Ticket` or `ticket_data`
- category label: `Category` or `label`

## Run the Streamlit App

```bash
streamlit run app.py
```

The app expects the prebuilt index and metadata file.

## Implementation Details

- Representative index: samples per category are embedded and averaged to build
  one vector per category, then indexed with FAISS cosine similarity.
- Retrieval: each ticket is embedded and used to retrieve the top 3 categories.
- Reranking: a cross-encoder re-scores the top 3 and produces a hybrid score.
- Hybrid scoring: similarity is normalized to $[0,1]$, rerank scores are
  normalized to $[-1,1]$, and combined with weights (default 0.9 similarity,
  0.1 rerank).
- Rate-limit handling: Gemini calls retry on 429 RESOURCE_EXHAUSTED with
  exponential backoff.
- Cross-encoder device: auto-falls back to CPU if CUDA is unavailable.

## Streamlit Workflow and Features

- Load Sample: loads a random ticket from the dataset into the input box.
- Classify: runs cosine retrieval + reranking and shows the top 3 categories.
- View Confidence: displays the final hybrid score in green or red based on a
  threshold (default 0.85) and flags low-confidence tickets for human review.
- Generate Response: sends the ticket and predicted category to Gemini 2.5 Flash
  to draft a support response.
- Evaluation: supports similarity-only, similarity+reranking, and LLM (bulk)
  evaluation modes.

## Approach and Why

This uses a two-stage retrieval + reranking pipeline to balance speed and
accuracy. Representative category embeddings provide fast coarse retrieval, and
the cross-encoder adds precision on the short list without the cost of scoring
every category. The hybrid score preserves similarity strength while allowing
rerank scores to improve or reduce confidence based on sign and magnitude.

## How to Evaluate

- Offline: run the evaluation modes in the app or CLI to track accuracy, macro
  F1, weighted F1, and confusion matrix.
- Online: track human overrides, response time, and resolution outcomes for
  model quality and business impact.

CLI example:

```bash
python -c "from pathlib import Path; from classifier import run_evaluation; result = run_evaluation(Path('classified_support_tickets.csv'), sample_per_category=10); print(result.accuracy)"
```

## Production Changes for 10K Tickets/Month

- Hosting: deploy the index and model services behind an API; add caching for
  repeated tickets or categories.
- Scaling: precompute embeddings for known tickets and batch new ones; move
  embeddings to a managed vector store if needed.
- Monitoring: log confidence, latency, and human override rate; alert on drift.
- Quality: add active learning loops to update samples per category and retrain
  the representative index on a schedule.
- Reliability: implement request queues, retries, and fallbacks for LLM outages.

## Rough Cost Estimate (10K Tickets/Month)

Costs depend on current Gemini pricing and average token counts. A rough way to
estimate:

- Embeddings: tokens per ticket _ 10K / 1M _ embedding price per 1M tokens.
- Response generation: input+output tokens per ticket _ 10K / 1M _ model price.

Example sizing (replace with current prices): if a ticket averages 300 input
tokens and a response averages 150 output tokens, total model tokens are roughly
4.5M per month. Multiply 4.5M by the current per-1M-token prices for embeddings
and Gemini 2.5 Flash to get a monthly estimate. In most cases, this is in the
low single-digit to low double-digit USD range at 10K tickets/month, but check
the current Vertex AI pricing page to confirm.
