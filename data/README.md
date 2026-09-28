# Data

The data was provided by ESILV for the Information Retrieval & NLP course (2025-2026). It is
scraped from TripAdvisor and is **not redistributed** in this repository.

To run `tripreco evaluate`, put these two files in `data/raw/`:

| File | Content | Columns used |
|---|---|---|
| `Tripadvisor.csv` | 3 761 places | `id`, `nom`, `typeR`, `priceRange`, `restaurantType`, `restaurantTypeCuisine`, `activiteSubCategorie`, `activiteSubType` |
| `reviews83325.csv` | 340 385 reviews | `idplace`, `review`, `langue` |

`typeR` is `H` (hotel), `R` (restaurant), `A` (attraction) or `AP` (attraction product). The
category columns hold comma-separated ids. Only `review` is given to the models; the other
columns are used for evaluation only.

The tests use a small synthetic dataset with the same columns (`tests/conftest.py`).
