"""
CLI entry point.

Examples:
  python main.py --ingest data/sample.txt --query "What is this document about?"
  python main.py --ingest data/report.pdf data/notes.md --k 5 --query "Summarize section 2"
"""

import argparse
from dotenv import load_dotenv
from src.pipeline import RAGPipeline

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Basic RAG pipeline (from scratch)")
    parser.add_argument("--ingest", nargs="+", required=True, help="Path(s) to document(s)")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    pipeline = RAGPipeline(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    pipeline.ingest_many(args.ingest)

    answer, retrieved = pipeline.query(args.query, k=args.k)

    print("\n=== Retrieved Chunks ===")
    for i, (text, meta, score) in enumerate(retrieved):
        print(f"\n[{i + 1}] score={score:.3f} source={meta.get('source')}")
        preview = text[:300] + ("..." if len(text) > 300 else "")
        print(preview)

    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
