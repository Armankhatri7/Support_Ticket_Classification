# Support_Ticket_Classification

Support ticket classifier using Gemini embeddings and a FAISS index over
representative embeddings per category.

## Method

- Randomly sample 10 tickets per category.
- Embed the ticket content and average to a representative vector per category.
- Build a FAISS cosine-similarity index from those representative vectors.
- Embed each ticket and retrieve the top 3 representative categories.
- Re-rank the top 3 with a cross-encoder and pick the best as the prediction.
- Evaluate on a random sample using accuracy, macro F1, weighted F1, and a
  confusion matrix.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Note: The cross-encoder uses `sentence-transformers` (requires PyTorch). If the
install pulls CPU wheels, install a CUDA-enabled PyTorch build so the
cross-encoder runs on GPU.

Create a `.env` file (ignored by git) and add your key:

```bash
GEMINI_API_KEY=your_key_here
```

Note: If `faiss-cpu` fails to install on Windows, try:

```bash
conda install -c pytorch faiss-cpu
```

## Run the Streamlit app

```bash
streamlit run app.py
```

The app expects a prebuilt index and metadata file.

## Build the representative index

```bash
python build_representative_index.py --samples-per-category 10 --seed 42
```

Ensure `GEMINI_API_KEY` is set in `.env` before running the build.

Supported dataset column names:
- ticket text: `Ticket` or `ticket_data`
- category label: `Category` or `label`

This writes:

- `faiss_category.index`
- `faiss_category.meta.json`

## Run evaluation from the CLI

```bash
python -c "from pathlib import Path; from classifier import run_evaluation; result = run_evaluation(Path('classified_support_tickets.csv'), sample_per_category=10); print(result.accuracy)"
```
