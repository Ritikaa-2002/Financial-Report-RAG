"""Measure how often each retrieval setup finds the page that holds the answer.

For every FinanceBench question we know the filing and the page(s) where the
answer lives. A setup "hits" at k if one of its top-k retrieved pages is a gold
page. MRR@10 is the mean of 1/rank of the first gold page (0 if not in the top 10).

To avoid tuning on the test data, companies are split alphabetically:
the first 10 companies are the dev set (used to design the finance layer and
pick settings); the other 22 are the held-out test set reported in the README.

    python evaluate.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from report_rag.ingest import build_corpus
from report_rag.retrieve import METHODS, Retriever

DATA = Path("data")
RESULTS = Path("results")
KS = (1, 3, 5, 10)
N_DEV_COMPANIES = 10
CHUNKING = {"Whole page": dict(words_per_chunk=None), "200-word chunks": dict(words_per_chunk=200, overlap=50)}
LABELS = {
    "bm25": "BM25 (keyword)", "tfidf": "TF-IDF", "lsa": "LSA (dense)",
    "hybrid": "Hybrid (BM25 + LSA)", "finance": "Finance-aware hybrid",
}
ABLATIONS = {
    "Hybrid baseline": (False, False),
    "+ Query expansion only": (True, False),
    "+ Statement boost only": (False, True),
    "+ Both (finance-aware)": (True, True),
}


def load_questions():
    rows = [json.loads(l) for l in (DATA / "financebench_open_source.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        r["gold"] = {(e["doc_name"], int(e["evidence_page_num"])) for e in r["evidence"]}
    return rows


def split(questions):
    companies = sorted({q["company"] for q in questions})
    dev = set(companies[:N_DEV_COMPANIES])
    return [q for q in questions if q["company"] in dev], [q for q in questions if q["company"] not in dev]


def ranked_pages(hits):
    """Collapse a ranked list of chunks into a ranked list of unique (doc, page)."""
    seen, out = set(), []
    for chunk, _ in hits:
        key = (chunk.doc, chunk.page)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def first_gold_rank(pages, gold):
    for i, p in enumerate(pages, 1):
        if p in gold:
            return i
    return None


def summarize(ranks):
    n = len(ranks)
    ranks = [r if r == r else None for r in ranks]  # NaN -> None
    row = {f"Hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    row["MRR@10"] = sum(1 / r for r in ranks if r and r <= 10) / n
    return row


def run(retriever, questions, method, within_filing=True):
    return [
        first_gold_rank(
            ranked_pages(retriever.search(q["question"], k=60, method=method,
                                          doc=q["doc_name"] if within_filing else None)),
            q["gold"],
        )
        for q in questions
    ]


def main():
    RESULTS.mkdir(exist_ok=True)
    questions = load_questions()
    dev, test = split(questions)
    docs = sorted({q["doc_name"] for q in questions})
    print(f"{len(questions)} questions over {len(docs)} filings | dev {len(dev)} questions, test {len(test)} questions")

    rows, per_question, retrievers = [], [], {}
    for chunk_label, cfg in CHUNKING.items():
        corpus = build_corpus(DATA / "pdfs", docs=docs, **cfg)
        print(f"\n{chunk_label}: {len(corpus):,} chunks from {len({(c.doc, c.page) for c in corpus}):,} pages")
        retrievers[chunk_label] = r = Retriever(corpus)
        for scope, within in (("Within the filing", True), ("Across all filings", False)):
            for method in METHODS:
                for set_name, qs in (("dev", dev), ("test", test)):
                    ranks = run(r, qs, method, within)
                    rows.append({"Set": set_name, "Chunking": chunk_label, "Scope": scope,
                                 "Method": LABELS[method], **summarize(ranks)})
                    per_question += [{"financebench_id": q["financebench_id"], "set": set_name,
                                      "chunking": chunk_label, "scope": scope, "method": method,
                                      "question_type": q["question_type"], "pages_needed": len(q["gold"]),
                                      "rank": rk} for q, rk in zip(qs, ranks)]
                t = rows[-1]
                print(f"  {scope:19} {LABELS[method]:21} test Hit@1 {t['Hit@1']:.0%}  Hit@5 {t['Hit@5']:.0%}  MRR {t['MRR@10']:.2f}")

    metrics = pd.DataFrame(rows)
    metrics.round(3).to_csv(RESULTS / "retrieval_metrics.csv", index=False)
    pq = pd.DataFrame(per_question)
    pq.to_csv(RESULTS / "per_question_ranks.csv", index=False)

    # Ablation on the held-out test set
    r = retrievers["200-word chunks"]
    ablation = []
    for name, (expand, boost) in ABLATIONS.items():
        r.expand_query, r.statement_boost = expand, boost
        ablation.append({"Setup": name, **summarize(run(r, test, "finance"))})
    r.expand_query, r.statement_boost = True, True
    ablation = pd.DataFrame(ablation)
    ablation.round(3).to_csv(RESULTS / "ablation_test.csv", index=False)
    print("\nAblation (test set, 200-word chunks, within the filing):")
    print(ablation.round(3).to_string(index=False))

    # Breakdown: baseline hybrid vs finance-aware, test set
    sel = pq[(pq.set == "test") & (pq.chunking == "200-word chunks") & (pq.scope == "Within the filing")]
    breakdown = []
    for method in ("hybrid", "finance"):
        s = sel[sel.method == method].assign(
            pages_bucket=lambda d: d.pages_needed.map(lambda n: "1 page" if n == 1 else "2+ pages"))
        for col, label in (("question_type", "Question type"), ("pages_bucket", "Pages needed")):
            for value, g in s.groupby(col):
                breakdown.append({"Method": LABELS[method], "Group": label, "Value": value,
                                  "Questions": len(g), **summarize(list(g["rank"]))})
    breakdown = pd.DataFrame(breakdown)
    breakdown.round(3).to_csv(RESULTS / "breakdown_test.csv", index=False)
    print("\nBreakdown (test set):")
    print(breakdown.round(3).to_string(index=False))

    plot_methods(metrics)
    plot_ablation(ablation)
    plot_hit_at_k(metrics)


def plot_methods(metrics):
    t = metrics[(metrics.Set == "test") & (metrics.Scope == "Within the filing")]
    pivot = t.pivot(index="Method", columns="Chunking", values="Hit@5").loc[list(LABELS.values())]
    ax = pivot[list(CHUNKING)].plot(kind="barh", figsize=(9, 4.8), width=0.72, color=["#a9bfdc", "#2f5f98"])
    ax.set_xlabel("Hit@5: share of test questions with the answer page in the top 5")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, 1)
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.set_title("Retrieval accuracy by method (held-out companies, searching within the filing)", loc="left", fontsize=11.5)
    for c in ax.containers:
        ax.bar_label(c, labels=[f"{v:.0%}" for v in c.datavalues], padding=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULTS / "hit5_by_method.png", dpi=150)
    plt.close()


def plot_ablation(ablation):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = range(len(ablation))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], ablation["Hit@1"], w, label="Hit@1", color="#a9bfdc")
    b5 = ax.bar([i + w / 2 for i in x], ablation["Hit@5"], w, label="Hit@5", color="#2f5f98")
    for b in (b1, b5):
        ax.bar_label(b, labels=[f"{v:.0%}" for v in b.datavalues], padding=2, fontsize=9)
    ax.set_xticks(list(x), ablation["Setup"], fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("What each part of the finance-aware layer adds (test set)", loc="left", fontsize=11.5)
    plt.tight_layout()
    plt.savefig(RESULTS / "ablation.png", dpi=150)
    plt.close()


def plot_hit_at_k(metrics):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = metrics[(metrics.Set == "test") & (metrics.Chunking == "200-word chunks")]
    for scope in ("Within the filing", "Across all filings"):
        for method, color in (("Hybrid (BM25 + LSA)", "#a9bfdc"), ("Finance-aware hybrid", "#2f5f98")):
            g = t[(t.Scope == scope) & (t.Method == method)].iloc[0]
            ax.plot(KS, [g[f"Hit@{k}"] for k in KS], "-" if scope == "Within the filing" else "--",
                    marker="o", color=color, label=f"{method}, {scope.lower()}")
    ax.set_xticks(KS)
    ax.set_xlabel("Number of pages retrieved (k)")
    ax.set_ylabel("Hit@k")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Hit@k on held-out companies (200-word chunks)", loc="left", fontsize=11.5)
    ax.legend(frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(RESULTS / "hit_at_k.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
