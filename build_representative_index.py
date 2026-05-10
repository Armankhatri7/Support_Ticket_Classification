from __future__ import annotations

import argparse
from pathlib import Path

from classifier import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_PER_CATEGORY,
    build_representative_index_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a representative FAISS index from sampled tickets."
    )
    parser.add_argument(
        "--csv",
        default="classified_support_tickets.csv",
        help="Path to the ticket CSV.",
    )
    parser.add_argument(
        "--index",
        default="faiss_category.index",
        help="Output FAISS index path.",
    )
    parser.add_argument(
        "--meta",
        default="faiss_category.meta.json",
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=DEFAULT_SAMPLE_PER_CATEGORY,
        help="Number of ticket samples per category.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini embedding model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = build_representative_index_files(
        csv_path=Path(args.csv),
        index_path=Path(args.index),
        meta_path=Path(args.meta),
        sample_per_category=args.samples_per_category,
        seed=args.seed,
        model=args.model,
    )
    print(f"Saved index to {args.index}")
    print(f"Saved metadata to {args.meta}")
    print(f"Categories: {len(meta.get('categories', []))}")


if __name__ == "__main__":
    main()
