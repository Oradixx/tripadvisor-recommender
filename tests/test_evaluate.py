import numpy as np
import pandas as pd

from tripreco.evaluate import first_match_positions, random_expectation, relevance, summarise


def frame(rows):
    cols = ['typeR', 'priceRange', 'restaurantType', 'restaurantTypeCuisine', 'activiteSubCategorie', 'activiteSubType']
    return pd.DataFrame(rows, columns=cols)


Q = frame([['R', None, '10591', '5086,10654', None, None],
           ['A', None, None, None, '47', '163'],
           ['AP', None, None, None, None, None],
           ['H', '€€', None, None, None, None]])
C = frame([['R', None, '9999', '10654', None, None],      # shares one cuisine with query 0
           ['A', None, None, None, '26', '34'],            # attraction, other categories
           ['A', None, None, None, '26,47', None],         # shares sub-category 47 with query 1
           ['H', '€€€€', None, None, None, None]])


def test_level1_is_same_type():
    rel = relevance(Q, C, 1)
    assert rel.tolist() == [[True, False, False, False], [False, True, True, False],
                            [False, False, False, False], [False, False, False, True]]


def test_level2_needs_one_shared_category():
    rel = relevance(Q, C, 2)
    assert rel[0].tolist() == [True, False, False, False]
    assert rel[1].tolist() == [False, False, True, False]
    assert not rel[2].any()          # attraction products have no category: never testable
    assert not rel[3].any()          # different price range


def test_first_match_position_and_skipped_queries():
    rel = relevance(Q, C, 2)
    scores = np.array([[0.1, 0.9, 0.5, 0.0],    # query 0: its match (col 0) is ranked 3rd -> error 2
                       [0.0, 0.9, 0.8, 0.0],    # query 1: match (col 2) ranked 2nd -> error 1
                       [1, 1, 1, 1], [1, 1, 1, 1]], dtype=float)
    pos, keep = first_match_positions(scores, rel)
    assert keep.tolist() == [True, True, False, False]
    assert pos.tolist() == [2, 1]
    s = summarise(scores, rel)
    assert s['ranking_error'] == 1.5 and s['queries'] == 2 and s['hit_at_1'] == 0


def test_ties_are_broken_by_catalogue_order():
    rel = np.array([[False, True, True]])
    pos, _ = first_match_positions(np.zeros((1, 3)), rel)
    assert pos.tolist() == [1]


def test_random_expectation_matches_simulation():
    rng = np.random.default_rng(0)
    rel = np.zeros((1, 20), dtype=bool)
    rel[0, [3, 11, 17]] = True
    sims = [first_match_positions(rng.random((1, 20)), rel)[0][0] for _ in range(20000)]
    assert abs(np.mean(sims) - random_expectation(rel)) < 0.1     # (20 - 3) / (3 + 1) = 4.25
