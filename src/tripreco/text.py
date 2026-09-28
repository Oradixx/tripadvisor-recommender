import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"[a-z][a-z']+")


def whitespace(text):
    """The course version: lower case and split on spaces (punctuation stays glued to words)."""
    return str(text).lower().split()


def words(text):
    """Letters only, English stop words removed: 'museum.' and 'Museum' both become 'museum'."""
    return [w for w in _WORD.findall(str(text).lower()) if w not in ENGLISH_STOP_WORDS]
