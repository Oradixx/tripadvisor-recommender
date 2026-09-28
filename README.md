# TripAdvisor Recommender: similar places from reviews only

[![CI](https://github.com/Oradixx/tripadvisor-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/Oradixx/tripadvisor-recommender/actions/workflows/ci.yml)

Given the reviews of one place (a restaurant, a hotel, an attraction), recommend the most similar
place **using only what visitors wrote**. The place metadata (type, cuisine, price range,
attraction category) is never given to a model: it is only used afterwards to check whether the
recommendation was right.

This started as the first Information Retrieval & NLP project of our 4th year at ESILV (March
2026, [Clément Vurpillot](https://github.com/Oradixx) and Noé Spychala). The course version is kept
unchanged in [`notebooks/course_version.ipynb`](notebooks/course_version.ipynb) and
[`docs/course_report.pdf`](docs/course_report.pdf). The package in `src/` is a later rework (September
2026): a faster, tested evaluation, the missing measurements, several models, and a check of how
much of the results is noise.

## The task

- **Hypothesis**: similar experiences are described with similar words, so reviews alone should
  be enough to recommend a similar place.
- **Data** (provided by the course): 3 761 places and 340 385 reviews scraped from TripAdvisor.
  Only English reviews are used, which leaves **1 835 places**.
  - The data is not redistributed here (see [`data/README.md`](data/README.md)).
- **Protocol**: the places are split 50/50. Each place of the first half (917 queries) is used to
  rank all the places of the second half (918, the catalogue).
- **Ranking error**: the position (from 0) of the first catalogue place that matches the query.
  An error of 0 means the top recommendation is right. Queries with no possible match are skipped.
  - **Level 1**: same type (hotel, restaurant, attraction, attraction product).
  - **Level 2**: same type and at least one shared category. For hotels, the price range. For
    restaurants, the restaurant type or cuisine. For attractions, the sub-category or sub-type.
    That leaves 414 queries (attraction products have no category).

## Results

Official split of the brief (`random_state=42`), 10 reviews per place unless stated:

| Model | Test L1 | Test L2 | L2 median | L2 hit@1 |
|---|---:|---:|---:|---:|
| BM25, course baseline (`rank_bm25`, whitespace tokens) | **0.61** | 7.40 | 0 | 72 % |
| TF-IDF, course model (1 000 words) | 0.80 | 4.16* | 0 | 70 % |
| BM25 with a cleaner tokenizer | 0.62 | 3.37 | 0 | 73 % |
| TF-IDF, full vocabulary, sublinear tf | 0.66 | 3.46 | 0 | 72 % |
| Same with 30 reviews per place | 0.68 | **2.77** | 0 | 73 % |
| LSA (200 dimensions) | 0.70 | 3.67 | 0 | 73 % |
| Centroid of review vectors | 0.69 | 4.74 | 0 | 72 % |
| Fusion TF-IDF + centroid (**selected on validation**) | 0.68 | 3.96 | 0 | **74 %** |
| Random ranking (expected) | 3.43 | 30.43 | | |

\* 4.14 in the course notebook, 4.16 when rerun with current pandas and scikit-learn. The other
course numbers (0.61 and 0.80) are reproduced exactly.

Every place is compared with the 918 places of the catalogue. Full tables, per type and with
validation scores: [`results/results.md`](results/results.md) and
[`results/results.json`](results/results.json).

The single official split is noisy, so every model was also run on 20 other random 50/50 splits
(reporting only, not used to choose). Mean ± standard deviation, and the share of splits where
the model beats the course BM25 on the same split:

| Model | L1 | beats BM25 | L2 | beats BM25 |
|---|---:|---:|---:|---:|
| BM25, course baseline | 0.83 ± 0.18 | | 7.85 ± 2.59 | |
| TF-IDF, course model | 0.80 ± 0.18 | 60 % | 6.21 ± 1.57 | 80 % |
| BM25 with a cleaner tokenizer | **0.66 ± 0.12** | 90 % | **5.11 ± 1.81** | **100 %** |
| TF-IDF, full vocabulary, sublinear tf | 0.69 ± 0.10 | **95 %** | 5.26 ± 1.51 | 90 % |
| Fusion TF-IDF + centroid (selected) | 0.72 ± 0.12 | 85 % | 5.48 ± 1.55 | 80 % |

## What we learned

- **The hypothesis mostly holds.**
  - At level 1, the first recommendation has the right type 86 to 89 % of the time.
  - At level 2, the first recommendation has the right type and a shared category 70 to 74 % of
    the time.
  - A random ranking would put the first good place at position 3.4 (level 1) and 30 (level 2).
- **The course report missed a result.**
  - It only measured BM25 at level 1, where BM25 beat our TF-IDF.
  - Measured at level 2, BM25 makes an error of 7.40, against 4.16 for our TF-IDF. So the course
    model already beat the baseline on the stricter level.
- **The most reliable gain is tokenization.**
  - The course BM25 split on spaces only, so `museum`, `museum.` and `museum,` were three
    different words.
  - Keeping letters only and removing stop words makes the same BM25 better at level 2 on
    20 splits out of 20.
- **Beyond that, the differences are within noise.**
  - The official split happens to favour the course BM25 at level 1 (0.61 there, 0.83 on
    average).
  - On validation, the selected fusion model came first. On the other splits it is not better
    than a well-tokenized BM25 or a tuned TF-IDF.
  - The honest conclusion: several simple models are equivalent. The simplest one, BM25 with
    good tokens, is the one to keep, and the one `tripreco recommend` uses by default.
- **The mean hides the distribution.**
  - The median error is 0 for every model.
  - The level-2 mean is driven by a few attractions. The course BM25 makes an error of 20.7 on
    attractions, against 1.5 on restaurants.
  - Attractions are also where level 1 fails. With the cleaner BM25, 79 of the 121 wrong first
    recommendations are an attraction returned for an attraction product, or the reverse.
    A sight and a ticket or tour for that kind of sight are plausibly reviewed with the same words.
- **More reviews do not always help.**
  - Going from 10 to 30 reviews per place improves level 2 on the official split.
  - It does not help on validation.
  - With all the reviews, the largest places (up to 33 878 reviews) dominate the vocabulary.

## How it is built

- **BM25 as one sparse matrix product.**
  - `tripreco.models.BM25` computes Okapi BM25 for all queries at once. It gives the same scores
    as `rank_bm25.BM25Okapi.get_scores`, including its floor on negative IDF (checked by a test to
    1e-9).
  - 900 × 900 places take 0.6 s instead of 77 s, which is what makes the 20-split check cheap.
- **No leakage.** Only review text goes into the models. The metadata is used only in
  `tripreco.evaluate`. Models are fitted on the catalogue only.
- **Validation and test are separate.**
  - Settings and the final model were chosen on validation: three 50/50 re-splits of the query
    half only. The rule is the lowest level-2 error.
  - The catalogue half is only used for the final scores.
- **Deterministic ties.** Equal scores are broken by catalogue order. The course notebook used
  `argsort()[::-1]`, which broke ties the other way; it made no difference on the official split.
- **Tests**: 17 pytest tests on a small synthetic dataset with the same columns as the real files.
  They cover BM25 against `rank_bm25`, the level-1/level-2 rules, the skipped queries, the random
  expectation checked by simulation, and the end-to-end `evaluate` command. The tests run in CI.

```
src/tripreco/
  data.py       load the CSV files, English reviews, one text per place, 50/50 split
  text.py       tokenizers (course version, letters only without stop words)
  models.py     BM25, TF-IDF, LSA, review centroid, reciprocal rank fusion
  evaluate.py   level-1 / level-2 relevance, ranking error, median, hit@1, MRR, random baseline
  cli.py        tripreco evaluate / tripreco recommend
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
# put Tripadvisor.csv and reviews83325.csv in data/raw/ (see data/README.md)
tripreco evaluate            # about 2 minutes: validation, test, 20 random splits -> results/
tripreco recommend 188467    # the 5 places most similar to a place id
```

```
Place des Vosges (A) -> most similar places, from the reviews only:
  14113329  A    346.193  Statue de Louis XIII
   6642129  A    306.046  4ème arrondissement
  11770085  A    303.912  Square Barye
  14178491  A    285.267  Fontaines de la Place des Vosges
  11718629  A    279.812  Statue de Beaumarchais
```

## Limits and next steps

- The categories used to judge level 2 are coarse and sometimes noisy: a restaurant can share
  "European" with half of the catalogue, while an attraction can have a very specific sub-type.
- Part of the similarity is geographic, not about the experience. Reviews name streets and
  neighbourhoods: the closest places to the Place des Vosges are the statue and the fountains
  that stand on it.
- Only English reviews and the first reviews of each place in file order are used.
- Next: sentence embeddings (for example `all-MiniLM-L6-v2`) as another ranker, compared with the
  same 20-split protocol. Also an error analysis of the attraction queries.

## Authors

Clément Vurpillot and Noé Spychala. Course: Information Retrieval & NLP, ESILV, 2025-2026.
MIT licence (code only; the course data is not included).
