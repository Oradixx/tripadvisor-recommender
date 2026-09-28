"""Load the course data, keep English reviews and build one text per place.

Only the review text is ever given to a model. The place metadata (typeR, price range, cuisine,
attraction categories) is kept aside and used for evaluation only.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RS = '\x1e'        # ASCII record separator between the reviews of a place
SEP = f' {RS} '   # counts as whitespace for every tokenizer, so it never becomes a token

META_COLUMNS = ['id', 'nom', 'typeR', 'priceRange', 'restaurantType', 'restaurantTypeCuisine',
                'activiteSubCategorie', 'activiteSubType']


def load(data_dir):
    data_dir = Path(data_dir)
    reviews = pd.read_csv(data_dir / 'reviews83325.csv', usecols=['idplace', 'review', 'langue'],
                          dtype={'idplace': 'int64', 'review': 'string', 'langue': 'string'})
    places = pd.read_csv(data_dir / 'Tripadvisor.csv', usecols=META_COLUMNS, dtype='string')
    places['id'] = places['id'].astype('int64')
    return reviews, places


def place_texts(reviews, places, max_reviews=10, language='en'):
    """One document per place: its first `max_reviews` reviews in `language`, joined.

    Capping the number of reviews reduces the gap between places with 2 reviews and places
    with thousands (the "difficulty" of the brief). `max_reviews=None` keeps them all.
    """
    r = reviews[reviews['langue'] == language].dropna(subset=['review'])
    if max_reviews is not None:
        r = r.groupby('idplace', sort=False).head(max_reviews)
    docs = r.groupby('idplace')['review'].agg(SEP.join).rename('text').reset_index()
    out = places.merge(docs, left_on='id', right_on='idplace', how='inner').drop(columns='idplace')
    return out.reset_index(drop=True)


def split(df, seed=42):
    """50 % of places are queries, 50 % the catalogue to search, as in the brief."""
    queries, catalogue = train_test_split(df, test_size=0.5, random_state=seed)
    return queries.reset_index(drop=True), catalogue.reset_index(drop=True)
