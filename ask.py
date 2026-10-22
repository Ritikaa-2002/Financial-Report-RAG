"""Ask a question about a company filing and get a cited answer.

    python ask.py --list
    python ask.py --doc AMERICANEXPRESS_2022_10K "What was total revenue net of interest expense in 2022?"

With ANTHROPIC_API_KEY set, Claude writes an answer that cites page numbers.
Without it, you get the top passages and their pages, which is still useful
for finding where something is in a 150-page filing.
"""

import argparse
import os
import textwrap
from pathlib import Path

from report_rag.answer import DEFAULT_MODEL, generate_answer
from report_rag.ingest import build_corpus
from report_rag.retrieve import METHODS, Retriever

PDF_DIR = Path("data/pdfs")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?", help="Your question in plain English")
    p.add_argument("--doc", help="Filing to search, e.g. AMERICANEXPRESS_2022_10K (see --list)")
    p.add_argument("--list", action="store_true", help="List the available filings")
    p.add_argument("-k", type=int, default=5, help="Number of passages to retrieve")
    p.add_argument("--method", choices=METHODS, default="finance")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-llm", action="store_true", help="Only show retrieved passages")
    args = p.parse_args()

    available = sorted(f.stem for f in PDF_DIR.glob("*.pdf"))
    if not available:
        p.error("No filings found. Run: python scripts/download_data.py")
    if args.list or not args.question:
        print("\n".join(available))
        return
    if args.doc and args.doc not in available:
        p.error(f"Unknown filing {args.doc!r}. Use --list to see the options.")

    corpus = build_corpus(PDF_DIR, docs=[args.doc] if args.doc else None, words_per_chunk=200, overlap=50)
    retriever = Retriever(corpus)
    passages = retriever.search(args.question, k=args.k, method=args.method, doc=args.doc)

    print(f"\nTop {len(passages)} passages for: {args.question}\n")
    for i, (chunk, _score) in enumerate(passages, 1):
        print(f"[{i}] {chunk.doc}, page {chunk.page + 1}")
        print(textwrap.indent(textwrap.shorten(chunk.text, 400, placeholder=" ..."), "    "), "\n")

    if args.no_llm or not os.environ.get("ANTHROPIC_API_KEY"):
        if not args.no_llm:
            print("Set ANTHROPIC_API_KEY to get a written answer with citations.")
        return

    import anthropic

    print("Answer:\n")
    try:
        print(generate_answer(args.question, passages, model=args.model))
    except anthropic.APIError as e:
        print(f"Claude API error: {e}. The passages above are still the best matches.")


if __name__ == "__main__":
    main()
