"""tripreco evaluate: validation, model selection and final test scores, in one command.
tripreco recommend: the most similar places to a given place."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from . import data
from .evaluate import relevance, summarise
from .models import BM25, LSA, Centroid, Fusion, TfIdf

TUNED = {'max_features': None, 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.5}

# name -> (reviews kept per place, factory, description)
MODELS = {
    'bm25_course': (10, lambda: BM25(), 'BM25 baseline of the course (whitespace tokens, k1=1.5, b=0.75)'),
    'tfidf_course': (10, lambda: TfIdf(), 'Course model: TF-IDF, 1 000 words, English stop words'),
    'bm25_words': (10, lambda: BM25('words'), 'BM25 with a cleaner tokenizer (letters only, stop words removed)'),
    'tfidf_tuned': (10, lambda: TfIdf(**TUNED), 'TF-IDF on the full vocabulary, sublinear tf, min_df=2, max_df=0.5'),
    'tfidf_tuned_30': (30, lambda: TfIdf(**TUNED), 'Same, 30 reviews per place'),
    'lsa_200': (10, lambda: LSA(200, **TUNED), 'Tuned TF-IDF reduced to 200 latent dimensions (LSA)'),
    'centroid': (10, lambda: Centroid(**TUNED), 'Mean of the TF-IDF vectors of each review'),
    'fusion_tfidf_centroid': (10, lambda: Fusion([TfIdf(**TUNED), Centroid(**TUNED)]),
                              'Reciprocal rank fusion of tuned TF-IDF and centroid'),
}


def _score(model, queries, catalogue):
    return MODELS[model][1]().fit(catalogue['text']).score(queries['text'])


def robustness(texts_by_k, n_splits=20):
    """The brief's single 50/50 split is noisy (about 400 level-2 queries, the mean driven by a few
    attractions). Re-split all places `n_splits` times and report mean, spread and how often each
    model beats the course BM25 on the same split. Reporting only: selection is done on validation."""
    per = {n: {1: [], 2: []} for n in MODELS}
    for s in range(100, 100 + n_splits):
        cache = {}
        for name, (k, _, _) in MODELS.items():
            if k not in cache:
                q, c = data.split(texts_by_k[k], s)
                cache[k] = (q, c, relevance(q, c, 1), relevance(q, c, 2))
            q, c, r1, r2 = cache[k]
            scores = _score(name, q, c)
            per[name][1].append(summarise(scores, r1)['ranking_error'])
            per[name][2].append(summarise(scores, r2)['ranking_error'])
    base = per['bm25_course']
    return {n: {f'level{lvl}': {'mean': float(np.mean(v[lvl])), 'std': float(np.std(v[lvl])),
                               'beats_bm25_course': float(np.mean(np.array(v[lvl]) < np.array(base[lvl])))}
                for lvl in (1, 2)} for n, v in per.items()}


def evaluate(data_dir, out_dir, seed=42, val_seeds=(0, 1, 2), n_splits=20):
    reviews, places = data.load(data_dir)
    texts = {}
    results = {}
    for name, (k, _, desc) in MODELS.items():
        t0 = time.time()
        if k not in texts:
            texts[k] = data.place_texts(reviews, places, k)
        queries, catalogue = data.split(texts[k], seed)
        # validation: only inside the query half, the catalogue half is never looked at
        val = {1: [], 2: []}
        for s in val_seeds:
            vq, vc = (d.reset_index(drop=True) for d in train_test_split(queries, test_size=0.5, random_state=s))
            scores = _score(name, vq, vc)
            for level in (1, 2):
                val[level].append(summarise(scores, relevance(vq, vc, level))['ranking_error'])
        scores = _score(name, queries, catalogue)
        results[name] = {
            'description': desc, 'reviews_per_place': k,
            'validation': {f'level{lvl}': float(np.mean(v)) for lvl, v in val.items()},
            'test': {f'level{lvl}': summarise(scores, relevance(queries, catalogue, lvl), queries['typeR'])
                     for lvl in (1, 2)},
        }
        v, t = results[name]['validation'], results[name]['test']
        print(f"{name:24s} val L1 {v['level1']:.2f} L2 {v['level2']:.2f}"
              f" | test L1 {t['level1']['ranking_error']:.2f} L2 {t['level2']['ranking_error']:.2f}"
              f" ({time.time() - t0:.0f}s)", flush=True)
    selected = min((n for n in MODELS if not n.endswith('_course')), key=lambda n: results[n]['validation']['level2'])
    queries, catalogue = data.split(texts[10], seed)
    rob = robustness(texts, n_splits) if n_splits else {}
    for name, r in rob.items():
        results[name]['random_splits'] = r
    report = {'random_splits': n_splits, 'selected': selected,
              'selection_rule': 'lowest level-2 ranking error on validation',
              'places': len(queries) + len(catalogue), 'queries': len(queries), 'catalogue': len(catalogue),
              'models': results}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    (out / 'results.md').write_text(markdown(report))
    print(f'selected: {selected}')
    return report


def markdown(report):
    rows = [('| Model | Reviews / place | Val. L1 | Val. L2 | **Test L1** | **Test L2** | Test L2 median '
             '| Test L2 hit@1 |'),
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in report['models'].items():
        t1, t2 = r['test']['level1'], r['test']['level2']
        mark = ' (selected)' if name == report['selected'] else ''
        rows.append(f"| `{name}`{mark} | {r['reviews_per_place']} | {r['validation']['level1']:.2f} | "
                    f"{r['validation']['level2']:.2f} | {t1['ranking_error']:.2f} | {t2['ranking_error']:.2f} | "
                    f"{t2['median']:.0f} | {t2['hit_at_1']:.0%} |")
    any_model = next(iter(report['models'].values()))['test']
    random_row = (f"| random ranking (expected) | | | | {any_model['level1']['random_ranking_error']:.2f} | "
                  f"{any_model['level2']['random_ranking_error']:.2f} | | |")
    rows.append(random_row)
    if report.get('random_splits'):
        rows += ['', (f"Over {report['random_splits']} random 50/50 splits of all places (mean ± std, and share of "
                      "splits where the model beats the course BM25):"), '',
                 '| Model | L1 | L1 beats BM25 | L2 | L2 beats BM25 |', '|---|---:|---:|---:|---:|']
        for name, r in report['models'].items():
            a, b = r['random_splits']['level1'], r['random_splits']['level2']
            rows.append(f"| `{name}` | {a['mean']:.2f} ± {a['std']:.2f} | {a['beats_bm25_course']:.0%} | "
                        f"{b['mean']:.2f} ± {b['std']:.2f} | {b['beats_bm25_course']:.0%} |")
    return '\n'.join(rows) + '\n'


def recommend(data_dir, place_id, model, top):
    reviews, places = data.load(data_dir)
    k = MODELS[model][0]
    df = data.place_texts(reviews, places, k)
    query = df[df['id'] == place_id]
    if query.empty:
        raise SystemExit(f'place {place_id} has no English review')
    catalogue = df[df['id'] != place_id].reset_index(drop=True)
    scores = _score(model, query, catalogue)[0]
    q = query.iloc[0]
    print(f"{q['nom']} ({q['typeR']}) -> most similar places, from the reviews only:")
    for i in np.argsort(-scores, kind='stable')[:top]:
        row = catalogue.iloc[i]
        print(f"{row['id']:>10}  {row['typeR']:<3} {scores[i]:8.3f}  {row['nom']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog='tripreco')
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('evaluate', help='validation + test scores of every model')
    e.add_argument('--data-dir', default='data/raw')
    e.add_argument('--out', default='results')
    e.add_argument('--splits', type=int, default=20, help='random splits for the robustness check (0 = skip)')
    r = sub.add_parser('recommend', help='most similar places to a place id')
    r.add_argument('place_id', type=int)
    r.add_argument('--data-dir', default='data/raw')
    r.add_argument('--model', default='bm25_words', choices=sorted(MODELS),
                   help='default: the simplest of the equivalent models (see README)')
    r.add_argument('--top', type=int, default=5)
    a = p.parse_args(argv)
    if a.cmd == 'evaluate':
        evaluate(a.data_dir, a.out, n_splits=a.splits)
    else:
        recommend(a.data_dir, a.place_id, a.model, a.top)


if __name__ == '__main__':
    main()
