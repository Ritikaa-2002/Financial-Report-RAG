"""Retrieval methods: BM25 keyword search, TF-IDF, LSA (dense vectors) and a hybrid.

Everything runs locally with scikit-learn, so no model downloads or API keys are
needed to search. Each method returns the top-k chunks for a question, optionally
restricted to a single filing.
"""

import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .finance_query import analyze, page_statements

METHODS = ("bm25", "tfidf", "lsa", "hybrid", "finance")


def tokenize(text):
    """Lowercase words and numbers; split 'FY2018' into 'fy' and '2018'."""
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]


class Retriever:
    def __init__(self, chunks, lsa_dims=256, bm25_k1=1.5, bm25_b=0.75, random_state=0, statement_weight=0.5,
                 expand_query=True, statement_boost=True):
        self.chunks = chunks
        self.doc_of = np.array([c.doc for c in chunks])
        self.statement_weight = statement_weight
        self.expand_query = expand_query
        self.statement_boost = statement_boost

        # Label every chunk with the financial statement its page shows (if any),
        # using the header text at the top of the page.
        page_label = {}
        for c in chunks:
            if (c.doc, c.page) not in page_label:
                page_label[(c.doc, c.page)] = page_statements(c.text)
        self.statements_of = [page_label[(c.doc, c.page)] for c in chunks]
        texts = [c.text for c in chunks]

        # BM25 on raw term counts
        self._counter = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self._counter.fit_transform(texts).tocsc().astype(np.float32)
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        df = np.diff(tf.indptr)
        self._idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocsr()
        denom_add = bm25_k1 * (1 - bm25_b + bm25_b * doc_len / doc_len.mean())
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tf.data = tf.data * (bm25_k1 + 1) / (tf.data + denom_add[rows])
        self._bm25 = tf.tocsc()

        # TF-IDF and its LSA projection (dense "semantic" vectors)
        self._tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                      sublinear_tf=True, min_df=2)
        tfidf_matrix = self._tfidf.fit_transform(texts)
        self._tfidf_matrix = tfidf_matrix
        dims = min(lsa_dims, tfidf_matrix.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=dims, random_state=random_state)
        self._lsa = normalize(self._svd.fit_transform(tfidf_matrix))

    # ----- scoring -----
    def _scores_bm25(self, query):
        vocab = self._counter.vocabulary_
        cols = [vocab[t] for t in tokenize(query) if t in vocab]
        if not cols:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.asarray(self._bm25[:, cols] @ self._idf[cols]).ravel()

    def _scores_tfidf(self, query):
        q = self._tfidf.transform([query])
        return np.asarray((self._tfidf_matrix @ q.T).todense()).ravel()

    def _scores_lsa(self, query):
        q = normalize(self._svd.transform(self._tfidf.transform([query])))
        return self._lsa @ q.ravel()

    def scores(self, query, method, mask=None):
        """Score every chunk. mask (bool array) limits scoring to some chunks, e.g. one filing."""
        restrict = (lambda s: np.where(mask, s, -np.inf)) if mask is not None else (lambda s: s)
        if method == "hybrid":
            return self._rrf([restrict(self._scores_bm25(query)), restrict(self._scores_lsa(query))])
        if method == "finance":
            return self._scores_finance(query, restrict)
        raw = {"bm25": self._scores_bm25, "tfidf": self._scores_tfidf, "lsa": self._scores_lsa}[method](query)
        return restrict(raw.astype(np.float64))

    def _scores_finance(self, query, restrict):
        """Hybrid search on a finance-expanded query, boosting the statements it needs."""
        expanded, needed = analyze(query)
        if not self.expand_query:
            expanded = query
        base = self._rrf([restrict(self._scores_bm25(expanded)), restrict(self._scores_lsa(expanded))])
        if not needed or not self.statement_boost:
            return base
        on_statement = np.array([bool(s & needed) for s in self.statements_of])
        statement_list = np.where(on_statement & np.isfinite(base) & (base > 0), base, -np.inf)
        return self._rrf([base, statement_list], weights=[1.0, self.statement_weight])

    @staticmethod
    def _rrf(score_lists, k=60, weights=None):
        """Reciprocal rank fusion: reward chunks ranked highly by any list.

        Items scored -inf in a list (not part of it) get nothing from that list.
        """
        weights = weights or [1.0] * len(score_lists)
        fused = np.zeros(len(score_lists[0]), dtype=np.float64)
        for s, w in zip(score_lists, weights):
            ranks = np.empty(len(s), dtype=np.int64)
            ranks[np.argsort(-s, kind="stable")] = np.arange(len(s))
            fused += np.where(np.isfinite(s), w / (k + ranks + 1), 0.0)
        return fused

    # ----- public API -----
    def search(self, query, k=5, method="hybrid", doc=None):
        """Return the top-k (chunk, score) pairs, optionally only from one filing."""
        mask = (self.doc_of == doc) if doc is not None else None
        s = self.scores(query, method, mask).astype(np.float64)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        top = np.argpartition(-s, min(k, len(s) - 1))[: k * 4]
        top = top[np.argsort(-s[top], kind="stable")]
        return [(self.chunks[i], float(s[i])) for i in top if np.isfinite(s[i])][:k]
