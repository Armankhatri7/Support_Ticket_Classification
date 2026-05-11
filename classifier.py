from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

TICKET_COL = "Ticket"
CATEGORY_COL = "Category"
TICKET_COL_ALIASES = ("Ticket", "ticket_data")
CATEGORY_COL_ALIASES = ("Category", "label")
DEFAULT_MODEL = "models/gemini-embedding-001"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CROSS_ENCODER_DEVICE = "cuda"
DEFAULT_SAMPLE_PER_CATEGORY = 10
DEFAULT_RERANK_TOP_K = 3
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_HYBRID_SIM_WEIGHT = 0.8
DEFAULT_HYBRID_RERANK_WEIGHT = 0.2
ENV_PATH = Path(__file__).with_name(".env")


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return "RESOURCE_EXHAUSTED" in message or " 429" in message or "429 " in message


def _retry_call(func, *, max_retries: int = 3, base_delay: float = 1.0):
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            delay = (base_delay * (2 ** attempt)) + (random.random() * 0.25)
            time.sleep(delay)


@dataclass
class EvaluationResult:
    sample_size: int
    categories: List[str]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    classification_report: str
    predictions_df: pd.DataFrame
    confusion_df: pd.DataFrame
    per_category_df: pd.DataFrame
    avg_similarity_correct: float
    avg_similarity_incorrect: float
    avg_rerank_correct: Optional[float] = None
    avg_rerank_incorrect: Optional[float] = None
    latency_seconds: float = 0.0


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self.model = model
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def embed_texts(
        self,
        texts: Iterable[str],
        task_type: str,
        batch_size: int = 16,
    ) -> np.ndarray:
        text_list = list(texts)
        embeddings: List[Optional[np.ndarray]] = [None] * len(text_list)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, text in enumerate(text_list):
            key = (text, task_type)
            if key in self._cache:
                embeddings[idx] = self._cache[key]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            for start in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[start : start + batch_size]
                response = _retry_call(
                    lambda: self._client.models.embed_content(
                        model=self.model,
                        contents=batch_texts,
                    )
                )
                batch_embeddings = response.embeddings or []
                if len(batch_embeddings) != len(batch_texts):
                    raise ValueError("Embedding batch size mismatch.")
                for offset, emb in enumerate(batch_embeddings):
                    vector = np.asarray(emb.values, dtype="float32")
                    text_idx = uncached_indices[start + offset]
                    key = (text_list[text_idx], task_type)
                    self._cache[key] = vector
                    embeddings[text_idx] = vector

        if any(item is None for item in embeddings):
            raise ValueError("Missing embeddings for one or more texts.")
        return np.vstack(embeddings).astype("float32")


def resolve_api_key(explicit_key: Optional[str]) -> str:
    load_dotenv(ENV_PATH)
    if explicit_key:
        return explicit_key
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    ticket_col = next((col for col in TICKET_COL_ALIASES if col in df.columns), None)
    category_col = next((col for col in CATEGORY_COL_ALIASES if col in df.columns), None)
    if ticket_col is None or category_col is None:
        raise ValueError(
            f"CSV must include ticket/category columns. Supported pairs: "
            f"{TICKET_COL_ALIASES!r} and {CATEGORY_COL_ALIASES!r}."
        )
    if ticket_col != TICKET_COL or category_col != CATEGORY_COL:
        df = df.rename(columns={ticket_col: TICKET_COL, category_col: CATEGORY_COL})

    df = df.dropna(subset=[TICKET_COL, CATEGORY_COL]).copy()
    df[TICKET_COL] = df[TICKET_COL].astype(str)
    df[CATEGORY_COL] = df[CATEGORY_COL].astype(str).str.strip()
    df = df[df[TICKET_COL].str.len() > 0]
    df = df[df[CATEGORY_COL].str.len() > 0]
    return df


def get_categories(df: pd.DataFrame) -> List[str]:
    return sorted(df[CATEGORY_COL].unique().tolist())


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def create_cross_encoder(
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    device: str = DEFAULT_CROSS_ENCODER_DEVICE,
) -> CrossEncoder:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return CrossEncoder(model_name, device=device)


def rerank_candidates(
    ticket_text: str,
    candidate_categories: List[str],
    cross_encoder: CrossEncoder,
) -> List[float]:
    pairs = [(ticket_text, category) for category in candidate_categories]
    scores = cross_encoder.predict(pairs)
    return [float(score) for score in scores]


def _normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5 for _ in values]
    return [(val - min_val) / (max_val - min_val) for val in values]


def _normalize_prediction(
    prediction: str, category_lookup: Dict[str, str]
) -> str:
    cleaned = prediction.strip()
    if not cleaned:
        return "UNKNOWN"
    normalized = category_lookup.get(cleaned.lower())
    return normalized or "UNKNOWN"


def llm_bulk_classify(
    tickets: List[str],
    categories: List[str],
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
) -> List[str]:
    if not api_key:
        raise ValueError("Gemini API key is required.")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )
    category_list = ", ".join(categories)
    ticket_lines = "\n".join(
        f"{idx + 1}. {text.strip()}" for idx, text in enumerate(tickets)
    )
    prompt = (
        "You are a support ticket classifier.\n"
        "Pick the single best category from the list for each ticket.\n"
        f"Categories: {category_list}\n"
        "Return JSON only, with the exact structure:\n"
        "{\"predictions\": [\"Category1\", \"Category2\", ...]}\n"
        "Use the same order as the tickets and only categories from the list.\n"
        "Tickets:\n"
        f"{ticket_lines}"
    )

    response = _retry_call(
        lambda: client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
    )
    raw_text = (response.text or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("` ")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    payload: Any
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response was not valid JSON.") from exc

    if isinstance(payload, list):
        predictions = payload
    else:
        predictions = payload.get("predictions") if isinstance(payload, dict) else None

    if not isinstance(predictions, list) or len(predictions) != len(tickets):
        raise ValueError("LLM response does not contain the expected predictions.")

    category_lookup = {category.lower(): category for category in categories}
    normalized = [
        _normalize_prediction(str(pred), category_lookup) for pred in predictions
    ]
    return normalized


def build_representative_index(
    df: pd.DataFrame,
    categories: List[str],
    embedder: GeminiEmbedder,
    sample_per_category: int = DEFAULT_SAMPLE_PER_CATEGORY,
    seed: int = 42,
    index_path: Optional[Path] = None,
) -> Tuple[faiss.IndexFlatIP, np.ndarray, Dict[str, int]]:
    vectors: List[np.ndarray] = []
    counts: Dict[str, int] = {}

    for category in categories:
        subset = df[df[CATEGORY_COL] == category]
        if subset.empty:
            raise ValueError(f"No samples found for category: {category}")
        sample_count = min(sample_per_category, len(subset))
        sample_df = subset.sample(n=sample_count, random_state=seed)
        ticket_texts = sample_df[TICKET_COL].tolist()
        embeddings = embedder.embed_texts(
            ticket_texts, task_type="retrieval_document"
        )
        vectors.append(embeddings.mean(axis=0))
        counts[category] = sample_count

    representative_vectors = np.vstack(vectors).astype("float32")
    representative_vectors = l2_normalize(representative_vectors)

    index = faiss.IndexFlatIP(representative_vectors.shape[1])
    index.add(representative_vectors)
    if index_path:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
    return index, representative_vectors, counts


def save_index_metadata(
    meta_path: Path,
    categories: List[str],
    counts: Dict[str, int],
    sample_per_category: int,
    seed: int,
    model: str,
    dataset_rows: int,
) -> Dict[str, Any]:
    meta = {
        "categories": categories,
        "counts": counts,
        "sample_per_category": sample_per_category,
        "seed": seed,
        "model": model,
        "dataset_rows": dataset_rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    return meta


def load_representative_index(
    index_path: Path, meta_path: Path
) -> Tuple[faiss.IndexFlatIP, List[str], Dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    categories = meta.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("Metadata file missing categories list.")
    return index, categories, meta


def build_representative_index_files(
    csv_path: Path,
    index_path: Path,
    meta_path: Path,
    sample_per_category: int = DEFAULT_SAMPLE_PER_CATEGORY,
    seed: int = 42,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        raise ValueError("Gemini API key missing. Set GEMINI_API_KEY in .env.")
    df = load_dataset(csv_path)
    categories = get_categories(df)
    embedder = GeminiEmbedder(resolved_key, model=model)
    _, _, counts = build_representative_index(
        df,
        categories,
        embedder,
        sample_per_category=sample_per_category,
        seed=seed,
        index_path=index_path,
    )
    return save_index_metadata(
        meta_path=meta_path,
        categories=categories,
        counts=counts,
        sample_per_category=sample_per_category,
        seed=seed,
        model=model,
        dataset_rows=len(df),
    )


def classify_tickets(
    tickets: List[str],
    categories: List[str],
    index: faiss.IndexFlatIP,
    embedder: GeminiEmbedder,
    top_k: int = 1,
    rerank: bool = False,
    cross_encoder: Optional[CrossEncoder] = None,
    hybrid_sim_weight: float = DEFAULT_HYBRID_SIM_WEIGHT,
    hybrid_rerank_weight: float = DEFAULT_HYBRID_RERANK_WEIGHT,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
) -> Tuple[List[str], List[float], Optional[List[float]], np.ndarray, np.ndarray]:
    if rerank and cross_encoder is None:
        raise ValueError("cross_encoder is required when rerank=True.")
    retrieval_k = max(top_k, rerank_top_k) if rerank else top_k
    ticket_vectors = embedder.embed_texts(tickets, task_type="retrieval_query")
    ticket_vectors = l2_normalize(ticket_vectors)
    scores, indices = index.search(ticket_vectors, retrieval_k)

    if not rerank:
        top_indices = indices[:, 0]
        predictions = [categories[idx] for idx in top_indices]
        top_scores = scores[:, 0].astype(float).tolist()
        return predictions, top_scores, None, scores, indices

    predictions: List[str] = []
    top_scores: List[float] = []
    top_rerank_scores: List[float] = []
    for row_idx, ticket_text in enumerate(tickets):
        candidate_indices = indices[row_idx][:rerank_top_k].tolist()
        candidate_categories = [categories[idx] for idx in candidate_indices]
        rerank_scores = rerank_candidates(
            ticket_text, candidate_categories, cross_encoder
        )
        similarity_scores = [float(scores[row_idx][pos]) for pos in range(rerank_top_k)]
        sim_norm = _normalize_scores(similarity_scores)
        rerank_norm = _normalize_scores(rerank_scores)
        hybrid_scores = [
            (hybrid_sim_weight * sim_val) + (hybrid_rerank_weight * rerank_val)
            for sim_val, rerank_val in zip(sim_norm, rerank_norm)
        ]
        best_pos = int(np.argmax(hybrid_scores))
        predictions.append(candidate_categories[best_pos])
        top_scores.append(float(hybrid_scores[best_pos]))
        top_rerank_scores.append(float(rerank_scores[best_pos]))
    return predictions, top_scores, top_rerank_scores, scores, indices


def score_ticket(
    ticket_text: str,
    categories: List[str],
    index: faiss.IndexFlatIP,
    embedder: GeminiEmbedder,
    top_k: int = 3,
    rerank: bool = False,
    cross_encoder: Optional[CrossEncoder] = None,
    hybrid_sim_weight: float = DEFAULT_HYBRID_SIM_WEIGHT,
    hybrid_rerank_weight: float = DEFAULT_HYBRID_RERANK_WEIGHT,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
) -> List[Dict[str, float]]:
    if rerank and cross_encoder is None:
        raise ValueError("cross_encoder is required when rerank=True.")
    _, _, scores, indices = classify_tickets(
        [ticket_text],
        categories,
        index,
        embedder,
        top_k=top_k,
        rerank=False,
    )
    candidate_indices = indices[0].tolist()
    candidate_categories = [categories[idx] for idx in candidate_indices]
    results: List[Dict[str, float]] = []

    if rerank:
        rerank_scores = rerank_candidates(
            ticket_text, candidate_categories, cross_encoder
        )
        similarity_scores = [float(scores[0][idx]) for idx in range(len(candidate_indices))]
        sim_norm = _normalize_scores(similarity_scores)
        rerank_norm = _normalize_scores(rerank_scores)
        hybrid_scores = [
            (hybrid_sim_weight * sim_val) + (hybrid_rerank_weight * rerank_val)
            for sim_val, rerank_val in zip(sim_norm, rerank_norm)
        ]
        for idx, category in enumerate(candidate_categories):
            results.append(
                {
                    "category": category,
                    "faiss_score": float(scores[0][idx]),
                    "rerank_score": float(rerank_scores[idx]),
                    "hybrid_score": float(hybrid_scores[idx]),
                }
            )
        results.sort(key=lambda row: row["hybrid_score"], reverse=True)
        return results

    for rank, idx in enumerate(candidate_indices):
        results.append({"category": categories[idx], "score": float(scores[0][rank])})
    return results


def evaluate_sample(
    df: pd.DataFrame,
    categories: List[str],
    index: faiss.IndexFlatIP,
    embedder: GeminiEmbedder,
    sample_size: int = 100,
    seed: int = 42,
    rerank: bool = True,
    cross_encoder: Optional[CrossEncoder] = None,
    hybrid_sim_weight: float = DEFAULT_HYBRID_SIM_WEIGHT,
    hybrid_rerank_weight: float = DEFAULT_HYBRID_RERANK_WEIGHT,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
) -> EvaluationResult:
    sample_size = min(sample_size, len(df))
    sample_df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    tickets = sample_df[TICKET_COL].tolist()
    y_true = sample_df[CATEGORY_COL].tolist()

    start_time = time.perf_counter()
    y_pred, scores, rerank_scores, _, _ = classify_tickets(
        tickets,
        categories,
        index,
        embedder,
        top_k=rerank_top_k if rerank else 1,
        rerank=rerank,
        cross_encoder=cross_encoder,
        hybrid_sim_weight=hybrid_sim_weight,
        hybrid_rerank_weight=hybrid_rerank_weight,
        rerank_top_k=rerank_top_k,
    )
    latency_seconds = time.perf_counter() - start_time

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=categories, zero_division=0
    )
    confusion = confusion_matrix(y_true, y_pred, labels=categories)

    predictions_df = sample_df[[TICKET_COL, CATEGORY_COL]].copy()
    predictions_df["Predicted"] = y_pred
    predictions_df["Similarity"] = scores
    if rerank_scores is not None:
        predictions_df["RerankScore"] = rerank_scores

    confusion_df = pd.DataFrame(confusion, index=categories, columns=categories)

    row_sums = confusion_df.sum(axis=1).replace(0, 1)
    per_category_df = pd.DataFrame(
        {
            "category": categories,
            "accuracy": np.diag(confusion_df.values) / row_sums.values,
            "support": row_sums.values,
        }
    )

    score_array = np.asarray(scores, dtype=float)
    correct_mask = np.array([yt == yp for yt, yp in zip(y_true, y_pred)])
    avg_similarity_correct = (
        float(score_array[correct_mask].mean()) if correct_mask.any() else 0.0
    )
    avg_similarity_incorrect = (
        float(score_array[~correct_mask].mean()) if (~correct_mask).any() else 0.0
    )

    avg_rerank_correct = None
    avg_rerank_incorrect = None
    if rerank_scores is not None:
        rerank_array = np.asarray(rerank_scores, dtype=float)
        avg_rerank_correct = (
            float(rerank_array[correct_mask].mean()) if correct_mask.any() else 0.0
        )
        avg_rerank_incorrect = (
            float(rerank_array[~correct_mask].mean()) if (~correct_mask).any() else 0.0
        )

    return EvaluationResult(
        sample_size=sample_size,
        categories=categories,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        classification_report=report,
        predictions_df=predictions_df,
        confusion_df=confusion_df,
        per_category_df=per_category_df,
        avg_similarity_correct=avg_similarity_correct,
        avg_similarity_incorrect=avg_similarity_incorrect,
        avg_rerank_correct=avg_rerank_correct,
        avg_rerank_incorrect=avg_rerank_incorrect,
        latency_seconds=float(latency_seconds),
    )


def evaluate_sample_llm(
    df: pd.DataFrame,
    categories: List[str],
    sample_size: int = 100,
    seed: int = 42,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
) -> EvaluationResult:
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        raise ValueError("Gemini API key missing. Set GEMINI_API_KEY in .env.")

    sample_size = min(sample_size, len(df))
    sample_df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    tickets = sample_df[TICKET_COL].tolist()
    y_true = sample_df[CATEGORY_COL].tolist()

    start_time = time.perf_counter()
    y_pred = llm_bulk_classify(
        tickets,
        categories,
        api_key=resolved_key,
        model=model,
    )
    latency_seconds = time.perf_counter() - start_time

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=categories, zero_division=0
    )
    confusion = confusion_matrix(y_true, y_pred, labels=categories)

    predictions_df = sample_df[[TICKET_COL, CATEGORY_COL]].copy()
    predictions_df["Predicted"] = y_pred

    confusion_df = pd.DataFrame(confusion, index=categories, columns=categories)
    row_sums = confusion_df.sum(axis=1).replace(0, 1)
    per_category_df = pd.DataFrame(
        {
            "category": categories,
            "accuracy": np.diag(confusion_df.values) / row_sums.values,
            "support": row_sums.values,
        }
    )

    return EvaluationResult(
        sample_size=sample_size,
        categories=categories,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        classification_report=report,
        predictions_df=predictions_df,
        confusion_df=confusion_df,
        per_category_df=per_category_df,
        avg_similarity_correct=0.0,
        avg_similarity_incorrect=0.0,
        avg_rerank_correct=None,
        avg_rerank_incorrect=None,
        latency_seconds=float(latency_seconds),
    )


def run_evaluation(
    csv_path: Path,
    sample_size: int = 100,
    sample_per_category: int = DEFAULT_SAMPLE_PER_CATEGORY,
    seed: int = 42,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    rerank: bool = True,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> EvaluationResult:
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        raise ValueError(
            "Gemini API key missing. Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    df = load_dataset(csv_path)
    categories = get_categories(df)
    embedder = GeminiEmbedder(resolved_key, model=model)
    cross_encoder = (
        create_cross_encoder(cross_encoder_model) if rerank else None
    )
    index, _, _ = build_representative_index(
        df,
        categories,
        embedder,
        sample_per_category=sample_per_category,
        seed=seed,
        index_path=None,
    )
    return evaluate_sample(
        df=df,
        categories=categories,
        index=index,
        embedder=embedder,
        sample_size=sample_size,
        seed=seed,
        rerank=rerank,
        cross_encoder=cross_encoder,
        rerank_top_k=rerank_top_k,
    )
