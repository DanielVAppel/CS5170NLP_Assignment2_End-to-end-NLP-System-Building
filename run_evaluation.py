"""
Usage:
    python run_evaluation.py                        # basic eval on test split
    python run_evaluation.py --split training       # basic eval on training split
    python run_evaluation.py --mode annotated       # annotated eval
    python run_evaluation.py --mode compare         # side-by-side comparison
    python run_evaluation.py --verbose              # print per-sample detail
"""

import argparse
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from Rag_Agent import get_vector_store, build_agent, ask, REVIEWS_XLSX, FAISS_INDEX
from Rag_evaluation import evaluate_basic, evaluate_annotated, compare_modes

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--split", choices=["test", "training"], default="test",
        help="Which data split to evaluate (default: test)"
    )
    parser.add_argument(
        "--mode", choices=["basic", "annotated", "compare"], default="basic",
        help="Evaluation mode (default: basic)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full per-sample breakdown after the summary"
    )
    parser.add_argument(
        "--rebuild-index", action="store_true",
        help="Force rebuild of the FAISS index from source xlsx files"
    )
    args = parser.parse_args()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

    print("[setup] Loading embeddings …")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
    )

    print("[setup] Loading vector store …")
    vector_store = get_vector_store(
        xlsx_path=REVIEWS_XLSX,
        embeddings=embeddings,
        index_path=FAISS_INDEX,
        force_rebuild=args.rebuild_index,
    )

    print("[setup] Building agent …")
    agent = build_agent(vector_store, openai_api_key)

    def query_fn(question: str) -> str:
        return ask(agent, question, stream=False)

    if args.mode == "basic":
        report = evaluate_basic(query_fn, split=args.split, verbose=True)
        if args.verbose:
            print(report.detailed())

    elif args.mode == "annotated":
        report = evaluate_annotated(query_fn, split=args.split, verbose=True)
        if args.verbose:
            print(report.detailed())

    elif args.mode == "compare":
        basic_report, annotated_report = compare_modes(query_fn, split=args.split)
        if args.verbose:
            print("\n── Basic detail ──")
            print(basic_report.detailed())
            print("\n── Annotated detail ──")
            print(annotated_report.detailed())


if __name__ == "__main__":
    main()