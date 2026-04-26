from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from Rag_Agent import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    FAISS_INDEX,
    REVIEWS_XLSX,
    TOP_K,
    ask,
    build_agent,
    get_vector_store,
)
from Rag_evaluation import compare_modes, evaluate_annotated, evaluate_basic

load_dotenv(override=True)


def _none_or_int(value: str) -> Optional[int]:
    if value.lower() in {"none", "null", "full"}:
        return None
    return int(value)


def _report_to_dict(report) -> dict:
    if hasattr(report, "to_dict"):
        return report.to_dict()
    return asdict(report)


def _save_json(data: dict, output_dir: Path, filename_prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{filename_prefix}_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")

    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--mode", choices=["basic", "annotated", "compare"], default="basic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-output-tokens", type=int, default=None)

    parser.add_argument(
        "--max-reviews",
        type=_none_or_int,
        default=None,
        help="Max review rows to load when building/rebuilding. Example: --max-reviews 1000 or --max-reviews full",
    )
    parser.add_argument(
        "--max-chunks",
        type=_none_or_int,
        default=None,
        help="Max chunks to embed when building/rebuilding. Example: --max-chunks 3000 or --max-chunks full",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=EMBEDDING_DIMENSIONS,
        help="Embedding dimensions. Must match the index dimensions. Default: 256",
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--output-dir", default="evaluation_results")

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be a positive integer.")
    if args.max_reviews is not None and args.max_reviews <= 0:
        raise ValueError("--max-reviews must be a positive integer or full.")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("--max-chunks must be a positive integer or full.")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

    print("[setup] Loading embeddings ...")
    print(f"[setup] embedding_dimensions={args.embedding_dimensions}")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=openai_api_key,
        dimensions=args.embedding_dimensions,
    )

    print("[setup] Loading vector store ...")
    print(f"[setup] rebuild_index={args.rebuild_index}")
    print(f"[setup] max_reviews={args.max_reviews}")
    print(f"[setup] max_chunks={args.max_chunks}")

    vector_store = get_vector_store(
        xlsx_path=REVIEWS_XLSX,
        embeddings=embeddings,
        index_path=FAISS_INDEX,
        force_rebuild=args.rebuild_index,
        max_reviews=args.max_reviews,
        max_chunks=args.max_chunks,
    )

    print("[setup] Building agent ...")
    print(f"[setup] top_k={args.top_k}, max_output_tokens={args.max_output_tokens}")

    agent = build_agent(
        vector_store=vector_store,
        openai_api_key=openai_api_key,
        top_k=args.top_k,
        max_output_tokens=args.max_output_tokens,
    )

    def query_fn(question: str) -> str:
        return ask(agent, question, stream=False)

    output_dir = Path(args.output_dir)

    if args.mode == "basic":
        report = evaluate_basic(query_fn, split=args.split, verbose=True, limit=args.limit)
        if args.verbose:
            print(report.detailed())
        if args.save_results:
            saved_path = _save_json(_report_to_dict(report), output_dir, f"basic_{args.split}")
            print(f"[save] Results saved to: {saved_path}")

    elif args.mode == "annotated":
        report = evaluate_annotated(query_fn, split=args.split, verbose=True, limit=args.limit)
        if args.verbose:
            print(report.detailed())
        if args.save_results:
            saved_path = _save_json(_report_to_dict(report), output_dir, f"annotated_{args.split}")
            print(f"[save] Results saved to: {saved_path}")

    elif args.mode == "compare":
        basic_report, annotated_report = compare_modes(query_fn, split=args.split, limit=args.limit)
        if args.verbose:
            print("\n-- Basic detail --")
            print(basic_report.detailed())
            print("\n-- Annotated detail --")
            print(annotated_report.detailed())

        if args.save_results:
            comparison = {
                "split": args.split,
                "basic": _report_to_dict(basic_report),
                "annotated": _report_to_dict(annotated_report),
            }
            saved_path = _save_json(comparison, output_dir, f"compare_{args.split}")
            print(f"[save] Comparison results saved to: {saved_path}")


if __name__ == "__main__":
    main()
