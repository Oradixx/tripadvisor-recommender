import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from tripreco.data import SEP
from tripreco.models import BM25, LSA, Centroid, Fusion, TfIdf
from tripreco.text import whitespace

DOCS = ['the museum has great paintings', 'great pizza and pasta, the pizza is the best',
        'clean room and nice breakfast', 'the park is great for a walk', 'pasta pasta pasta']
QUERIES = ['pizza pasta dinner', 'paintings museum art', 'the the great']


def test_bm25_matches_rank_bm25():
    ours = BM25().fit(DOCS).score(QUERIES)
    ref = BM25Okapi([whitespace(d) for d in DOCS])
    expected = np.vstack([ref.get_scores(whitespace(q)) for q in QUERIES])
    np.testing.assert_allclose(ours, expected, rtol=1e-9, atol=1e-12)


def test_bm25_binary_query_counts_each_term_once():
    once = BM25(binary_query=True).fit(DOCS).score(['pizza'])
    twice = BM25(binary_query=True).fit(DOCS).score(['pizza pizza'])
    np.testing.assert_allclose(once, twice)


@pytest.mark.parametrize('model', [TfIdf(), TfIdf(max_features=None, sublinear_tf=True),
                                   LSA(2, max_features=None), Centroid(max_features=None)])
def test_rankers_find_the_obvious_match(model):
    scores = model.fit(DOCS).score(['great pizza pasta'])
    assert scores.shape == (1, len(DOCS))
    assert np.argmax(scores[0]) == 1


def test_centroid_weights_reviews_equally():
    docs = [SEP.join(['pizza'] * 1 + ['museum paintings art gallery sculpture'] * 1), 'pizza']
    c = Centroid(max_features=None).fit(docs)
    # the first place is half pizza, half museum: its similarity to 'pizza' is about 1/sqrt(2)
    assert c.score(['pizza'])[0, 0] == pytest.approx(1 / np.sqrt(2), abs=1e-6)


def test_fusion_combines_ranks():
    f = Fusion([TfIdf(), BM25()]).fit(DOCS)
    assert np.argmax(f.score(['pizza pasta'])[0]) in (1, 4)
