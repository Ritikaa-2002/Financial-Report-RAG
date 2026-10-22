"""Download the FinanceBench open-source questions and the filings they refer to.

The data comes from the FinanceBench repository (Islam et al., 2023):
https://github.com/patronus-ai/financebench
It is downloaded at runtime rather than stored in this repo. The filings
themselves are public SEC filings and earnings releases.

    python scripts/download_data.py            # all 84 filings (~160 MB)
    python scripts/download_data.py --limit 5  # just a few, for a quick try
"""

import argparse
import json
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="Only download this many filings")
    args = p.parse_args()

    questions_path = DATA_DIR / "financebench_open_source.jsonl"
    fetch(f"{BASE}/data/financebench_open_source.jsonl", questions_path)
    questions = [json.loads(line) for line in questions_path.read_text().splitlines() if line.strip()]

    docs = sorted({q["doc_name"] for q in questions})
    if args.limit:
        docs = docs[: args.limit]

    for i, doc in enumerate(docs, 1):
        new = fetch(f"{BASE}/pdfs/{doc}.pdf", DATA_DIR / "pdfs" / f"{doc}.pdf")
        print(f"[{i}/{len(docs)}] {doc} {'downloaded' if new else 'already present'}")

    print(f"\n{len(questions)} questions, {len(docs)} filings in {DATA_DIR}")


if __name__ == "__main__":
    main()
