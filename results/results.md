| Model | Reviews / place | Val. L1 | Val. L2 | **Test L1** | **Test L2** | Test L2 median | Test L2 hit@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bm25_course` | 10 | 1.02 | 11.74 | 0.61 | 7.40 | 0 | 72% |
| `tfidf_course` | 10 | 0.81 | 6.37 | 0.80 | 4.16 | 0 | 70% |
| `bm25_words` | 10 | 0.64 | 8.31 | 0.62 | 3.37 | 0 | 73% |
| `tfidf_tuned` | 10 | 0.65 | 6.04 | 0.66 | 3.46 | 0 | 72% |
| `tfidf_tuned_30` | 30 | 0.58 | 6.57 | 0.68 | 2.77 | 0 | 73% |
| `lsa_200` | 10 | 0.80 | 6.85 | 0.70 | 3.67 | 0 | 73% |
| `centroid` | 10 | 0.70 | 5.89 | 0.69 | 4.74 | 0 | 72% |
| `fusion_tfidf_centroid` (selected) | 10 | 0.63 | 5.67 | 0.68 | 3.96 | 0 | 74% |
| random ranking (expected) | | | | 3.43 | 30.43 | | |

Over 20 random 50/50 splits of all places (mean ± std, and share of splits where the model beats the course BM25):

| Model | L1 | L1 beats BM25 | L2 | L2 beats BM25 |
|---|---:|---:|---:|---:|
| `bm25_course` | 0.83 ± 0.18 | 0% | 7.85 ± 2.59 | 0% |
| `tfidf_course` | 0.80 ± 0.18 | 60% | 6.21 ± 1.57 | 80% |
| `bm25_words` | 0.66 ± 0.12 | 90% | 5.11 ± 1.81 | 100% |
| `tfidf_tuned` | 0.69 ± 0.10 | 95% | 5.26 ± 1.51 | 90% |
| `tfidf_tuned_30` | 0.70 ± 0.14 | 80% | 5.17 ± 1.52 | 95% |
| `lsa_200` | 0.78 ± 0.12 | 55% | 5.21 ± 1.26 | 100% |
| `centroid` | 0.75 ± 0.16 | 75% | 5.88 ± 1.49 | 75% |
| `fusion_tfidf_centroid` | 0.72 ± 0.12 | 85% | 5.48 ± 1.55 | 80% |
