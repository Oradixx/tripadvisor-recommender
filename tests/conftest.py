"""A tiny synthetic TripAdvisor-like dataset with the columns of the course files.
The real data is not redistributed (see data/README.md)."""
import csv

import pytest

VOCAB = {
    ('R', '10591', '5086'): 'pasta pizza italian waiter dessert tiramisu wine dinner',
    ('R', '10591', '10654'): 'sushi ramen japanese chef fish rice lunch counter',
    ('H', '€€', None): 'room bed breakfast reception clean shower night stay',
    ('A', '47', '163'): 'museum paintings exhibition art gallery sculpture guide',
    ('A', '26', '34'): 'park garden trees walk lake flowers picnic',
    ('AP', None, None): 'tour bus ticket guide skip line booking group',
}


def write_sample(folder, places_per_kind=6, reviews_per_place=4):
    places, reviews = [], []
    pid, rid = 1000, 1
    for (kind, a, b), words in VOCAB.items():
        for _ in range(places_per_kind):
            pid += 1
            places.append({
                'id': pid, 'nom': f'{kind} place {pid}', 'typeR': kind,
                'priceRange': a if kind == 'H' else '',
                'restaurantType': a if kind == 'R' else '', 'restaurantTypeCuisine': b if kind == 'R' else '',
                'activiteSubCategorie': a if kind == 'A' else '', 'activiteSubType': b if kind == 'A' else '',
            })
            w = words.split()
            for j in range(reviews_per_place):
                text = ' '.join(w[(j + k) % len(w)] for k in range(5)) + ', really nice.'
                reviews.append({'idplace': pid, 'review': text.capitalize(), 'langue': 'en'})
                rid += 1
            reviews.append({'idplace': pid, 'review': 'Très bien, je recommande', 'langue': 'fr'})
    with open(folder / 'Tripadvisor.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(places[0]), quoting=csv.QUOTE_ALL)
        wr.writeheader()
        wr.writerows(places)
    with open(folder / 'reviews83325.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(reviews[0]), quoting=csv.QUOTE_ALL)
        wr.writeheader()
        wr.writerows(reviews)
    return folder


@pytest.fixture
def sample_dir(tmp_path):
    return write_sample(tmp_path)
