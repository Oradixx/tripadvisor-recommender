import json

from tripreco import cli, data


def test_place_texts_filters_language_and_caps_reviews(sample_dir):
    reviews, places = data.load(sample_dir)
    df = data.place_texts(reviews, places, max_reviews=2)
    assert len(df) == len(places)
    assert all(t.count(data.SEP) == 1 for t in df['text'])       # two reviews joined
    assert not df['text'].str.contains('Très').any()


def test_split_is_half_and_deterministic(sample_dir):
    df = data.place_texts(*data.load(sample_dir))
    q1, c1 = data.split(df)
    q2, _ = data.split(df)
    assert len(q1) == len(df) // 2 and len(q1) + len(c1) == len(df)
    assert q1['id'].tolist() == q2['id'].tolist()


def test_evaluate_end_to_end(sample_dir, tmp_path):
    report = cli.evaluate(sample_dir, tmp_path / 'out', n_splits=2)
    saved = json.loads((tmp_path / 'out' / 'results.json').read_text())
    assert saved['selected'] == report['selected'] and saved['selected'] in cli.MODELS
    for r in saved['models'].values():
        # the synthetic places are trivially separable by vocabulary
        assert r['test']['level1']['ranking_error'] < 1.0
        assert set(r['random_splits']) == {'level1', 'level2'}
    assert '| random ranking' in (tmp_path / 'out' / 'results.md').read_text()


def test_recommend_prints_places(sample_dir, capsys):
    cli.recommend(sample_dir, 1001, 'tfidf_tuned', 3)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith('R place 1001 (R)')
    assert len(lines) == 4 and all(' R ' in line for line in lines[1:])
