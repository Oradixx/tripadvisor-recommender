"""Ranking error of the brief.

For each query place, rank the whole catalogue; the error is the position (from 0) of the first
catalogue place that "matches" the query. Queries with no possible match in the catalogue are
skipped. Level 1: same `typeR`. Level 2: same type and at least one shared category (hotels: same
price range; restaurants: restaurant type or cuisine; attractions: sub-category or sub-type).
"""
import numpy as np
import pandas as pd


def _cats(value):
    if value is None or pd.isna(value):
        return frozenset()
    return frozenset(s.strip().lower() for s in str(value).split(',') if s.strip())


def level2_keys(df):
    keys = []
    for row in df.itertuples(index=False):
        t = row.typeR
        if t == 'H':
            cats = _cats(row.priceRange)
        elif t == 'R':
            cats = _cats(row.restaurantType) | _cats(row.restaurantTypeCuisine)
        elif t in ('A', 'AP'):
            cats = _cats(row.activiteSubCategorie) | _cats(row.activiteSubType)
        else:
            cats = frozenset()
        keys.append((t, cats))
    return keys


def relevance(queries, catalogue, level):
    """Boolean matrix (n_queries, n_catalogue): does catalogue place j match query i?"""
    if level == 1:
        q = queries['typeR'].to_numpy(dtype=object)
        c = catalogue['typeR'].to_numpy(dtype=object)
        return q[:, None] == c[None, :]
    kq, kc = level2_keys(queries), level2_keys(catalogue)
    return np.array([[tq == tc and len(cq & cc) > 0 for tc, cc in kc] for tq, cq in kq], dtype=bool)


def first_match_positions(scores, rel):
    """Position of the first relevant place in each query's ranking (ties broken by catalogue
    order, deterministic). Queries without any relevant place are dropped."""
    keep = rel.any(axis=1)
    order = np.argsort(-scores[keep], axis=1, kind='stable')
    hits = np.take_along_axis(rel[keep], order, axis=1)
    return hits.argmax(axis=1), keep


def random_expectation(rel):
    """Expected ranking error of a random ranking: with K relevant places among N, the first
    one is expected at position (N - K) / (K + 1)."""
    k = rel.sum(axis=1)
    n = rel.shape[1]
    k = k[k > 0]
    return float(np.mean((n - k) / (k + 1)))


def summarise(scores, rel, types=None):
    pos, keep = first_match_positions(scores, rel)
    out = {'ranking_error': float(pos.mean()), 'median': float(np.median(pos)),
           'hit_at_1': float((pos == 0).mean()), 'mrr': float(np.mean(1.0 / (pos + 1))),
           'queries': int(keep.sum()), 'random_ranking_error': random_expectation(rel)}
    if types is not None:
        t = np.asarray(types, dtype=object)[keep]
        out['by_type'] = {k: {'ranking_error': float(pos[t == k].mean()), 'queries': int((t == k).sum())}
                          for k in sorted(set(t))}
    return out
