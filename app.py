from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from classifier import (
    create_cross_encoder,
    evaluate_sample,
    evaluate_sample_llm,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_MODEL,
    DEFAULT_CROSS_ENCODER_DEVICE,
    DEFAULT_LLM_MODEL,
    GeminiEmbedder,
    load_dataset,
    load_representative_index,
    resolve_api_key,
    score_ticket,
)

INDEX_PATH = Path(__file__).with_name("faiss_category.index")
META_PATH = Path(__file__).with_name("faiss_category.meta.json")
DATA_PATH = Path(__file__).with_name("classified_support_tickets.csv")


@st.cache_resource(show_spinner=False)
def load_index(index_path: Path, meta_path: Path):
    return load_representative_index(index_path, meta_path)


def get_embedder(api_key: str, model: str) -> GeminiEmbedder:
    return GeminiEmbedder(api_key, model=model)


@st.cache_resource(show_spinner=False)
def get_cross_encoder(model_name: str):
    return create_cross_encoder(model_name, device=DEFAULT_CROSS_ENCODER_DEVICE)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    return load_dataset(path)


st.set_page_config(page_title="Support Ticket Classifier", layout="wide")

st.title("Support Ticket Classifier")
st.write("Representative-embedding classifier using sampled tickets per category.")
st.caption("Using GEMINI_API_KEY from .env")
st.caption(f"Cross-encoder device: {DEFAULT_CROSS_ENCODER_DEVICE}")

if not INDEX_PATH.exists() or not META_PATH.exists():
    st.error(
        "Representative index not found. Run `python build_representative_index.py` first."
    )
    st.stop()

try:
    index, categories, meta = load_index(INDEX_PATH, META_PATH)
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

model = str(meta.get("model") or DEFAULT_MODEL)
sample_per_category = int(meta.get("sample_per_category") or 10)
seed = int(meta.get("seed") or 42)
dataset_rows = meta.get("dataset_rows")

st.caption(
    f"Index model: {model} | Samples per category: {sample_per_category} | Seed: {seed}"
)

counts = meta.get("counts") or {}
if counts:
    counts_df = pd.DataFrame(
        {
            "category": categories,
            "samples": [counts.get(category, 0) for category in categories],
        }
    )
    st.subheader("Representative samples per category")
    st.dataframe(counts_df, use_container_width=True)

if dataset_rows is not None:
    st.caption(f"Dataset rows: {dataset_rows} | Categories: {len(categories)}")

st.divider()
st.subheader("Classify a ticket")
st.caption("Retrieves top 3 categories and re-ranks with a cross-encoder.")

user_text = st.text_area("Ticket text", height=180)
classify = st.button("Classify")

if classify:
    if not user_text.strip():
        st.error("Enter ticket text to classify.")
    else:
        resolved_key = resolve_api_key(None)
        if not resolved_key:
            st.error("Missing API key. Set GEMINI_API_KEY in .env.")
        else:
            with st.spinner("Embedding and searching..."):
                embedder = get_embedder(resolved_key, model)
                try:
                    cross_encoder = get_cross_encoder(DEFAULT_CROSS_ENCODER_MODEL)
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()
                top_matches = score_ticket(
                    user_text.strip(),
                    categories,
                    index,
                    embedder,
                    top_k=3,
                    rerank=True,
                    cross_encoder=cross_encoder,
                    rerank_top_k=3,
                )
            st.subheader("Top matches")
            st.dataframe(pd.DataFrame(top_matches), use_container_width=True)

st.divider()
st.subheader("Evaluation (100 random samples)")
st.caption(
    "Runs evaluation on 100 random tickets. LLM mode sends all tickets in one call."
)

df = load_data(DATA_PATH)
max_samples = len(df)
sample_size = min(100, max_samples)

eval_left, eval_right = st.columns(2)
with eval_left:
    eval_mode = st.selectbox(
        "Evaluation mode",
        (
            "Cosine similarity only",
            "Similarity + reranking",
            f"LLM ({DEFAULT_LLM_MODEL})",
        ),
    )
with eval_right:
    eval_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=100000,
        value=42,
        key="eval_seed",
    )

run_eval = st.button("Run evaluation")

if run_eval:
    resolved_key = resolve_api_key(None)
    if not resolved_key:
        st.error("Missing API key. Set GEMINI_API_KEY in .env.")
    else:
        with st.spinner("Evaluating..."):
            if eval_mode == "Cosine similarity only":
                embedder = get_embedder(resolved_key, model)
                results = evaluate_sample(
                    df,
                    categories,
                    index,
                    embedder,
                    sample_size=sample_size,
                    seed=int(eval_seed),
                    rerank=False,
                    cross_encoder=None,
                    rerank_top_k=3,
                )
            elif eval_mode == "Similarity + reranking":
                embedder = get_embedder(resolved_key, model)
                try:
                    cross_encoder = get_cross_encoder(DEFAULT_CROSS_ENCODER_MODEL)
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()
                results = evaluate_sample(
                    df,
                    categories,
                    index,
                    embedder,
                    sample_size=sample_size,
                    seed=int(eval_seed),
                    rerank=True,
                    cross_encoder=cross_encoder,
                    rerank_top_k=3,
                )
            else:
                try:
                    results = evaluate_sample_llm(
                        df,
                        categories,
                        sample_size=sample_size,
                        seed=int(eval_seed),
                        api_key=resolved_key,
                        model=DEFAULT_LLM_MODEL,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
        st.session_state["evaluation_results"] = results

evaluation_results = st.session_state.get("evaluation_results")

if evaluation_results:
    metrics_left, metrics_mid, metrics_right, metrics_extra = st.columns(4)
    metrics_left.metric("Accuracy", f"{evaluation_results.accuracy:.3f}")
    metrics_mid.metric("Macro F1", f"{evaluation_results.macro_f1:.3f}")
    metrics_right.metric("Weighted F1", f"{evaluation_results.weighted_f1:.3f}")
    metrics_extra.metric(
        "Latency (s)", f"{evaluation_results.latency_seconds:.2f}"
    )

    st.subheader("Evaluation predictions")
    st.dataframe(evaluation_results.predictions_df, use_container_width=True)
