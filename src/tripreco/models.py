"""Rankers: fit on the catalogue texts, score every query text against every catalogue place.

Every ranker returns a (n_queries, n_catalogue) score matrix, higher = more similar.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import RS
from .text import whitespace, words

TOKENIZERS = {'whitespace': whitespace, 'words': words}


class BM25:
    """Okapi BM25 as a sparse matrix product, identical to `rank_bm25.BM25Okapi.get_scores`
    (including its floor on negative IDF) but computed for all queries at once: ~1 s instead of
    ~80 s for 900 queries x 900 places.

    `binary_query=False` reproduces rank_bm25, which adds a term once per occurrence in the query;
    with long queries (a place's reviews) frequent words then dominate, so `True` counts each
    query term once.
    """

    def __init__(self, tokenizer='whitespace', k1=1.5, b=0.75, epsilon=0.25, binary_query=False):
        self.tokenizer, self.k1, self.b, self.epsilon, self.binary_query = tokenizer, k1, b, epsilon, binary_query

    def fit(self, texts):
        self.vec = CountVectorizer(tokenizer=TOKENIZERS[self.tokenizer], lowercase=False,
                                   token_pattern=None)
        tf = self.vec.fit_transform(texts).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        idf[idf < 0] = self.epsilon * idf.mean()
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        w = tf.copy()
        rows = np.repeat(np.arange(n_docs), np.diff(w.indptr))
        w.data = w.data * (self.k1 + 1) / (w.data + norm[rows])
        self.weights = (w @ sp.diags(idf)).T.tocsr()          # terms x docs
        return self

    def score(self, texts):
        q = self.vec.transform(texts).astype(np.float64)
        if self.binary_query:
            q.data[:] = 1.0
        return np.asarray((q @ self.weights).todense())


class TfIdf:
    """TF-IDF + cosine similarity. Defaults = the course version (1 000 words, English stop words)."""

    def __init__(self, **params):
        self.params = {'max_features': 1000, 'stop_words': 'english', **params}

    def fit(self, texts):
        self.vec = TfidfVectorizer(**self.params)
        self.doc = self.vec.fit_transform(texts)
        return self

    def score(self, texts):
        return np.asarray((self.vec.transform(texts) @ self.doc.T).todense())


class LSA(TfIdf):
    """TF-IDF projected on its first singular vectors: words that appear in the same places
    ('museum', 'exhibition', 'paintings') end up close, so two places can match without sharing
    the exact words."""

    def __init__(self, n_components=150, seed=0, **params):
        super().__init__(**params)
        self.n_components, self.seed = n_components, seed

    def fit(self, texts):
        super().fit(texts)
        n = min(self.n_components, self.doc.shape[1] - 1)     # small corpora (tests) have few words
        self.svd = TruncatedSVD(n, random_state=self.seed).fit(self.doc)
        self.doc_lsa = normalize(self.svd.transform(self.doc))
        return self

    def score(self, texts):
        return normalize(self.svd.transform(self.vec.transform(texts))) @ self.doc_lsa.T


class Fusion:
    """Reciprocal rank fusion of several rankers (each place gets sum of 1 / (k + rank))."""

    def __init__(self, rankers, k=60):
        self.rankers, self.k = rankers, k

    def fit(self, texts):
        for r in self.rankers:
            r.fit(texts)
        return self

    def score(self, texts):
        total = 0
        for r in self.rankers:
            ranks = np.argsort(np.argsort(-r.score(texts), axis=1, kind='stable'), axis=1)
            total = total + 1.0 / (self.k + ranks + 1)
        return total


class Centroid(TfIdf):
    """A place is the mean of its reviews' TF-IDF vectors instead of one long document, so one
    very long review cannot dominate the place, and every review counts the same."""

    def _embed(self, texts):
        reviews, owner = [], []
        for i, text in enumerate(texts):
            parts = [p for p in str(text).split(RS) if p.strip()] or ['']
            reviews += parts
            owner += [i] * len(parts)
        x = self.vec.transform(reviews)
        agg = sp.csr_matrix((np.ones(len(owner)), (owner, np.arange(len(owner)))), shape=(len(texts), len(owner)))
        return normalize(agg @ x)

    def fit(self, texts):
        super().fit(texts)
        self.doc = self._embed(texts)
        return self

    def score(self, texts):
        return np.asarray((self._embed(texts) @ self.doc.T).todense())
