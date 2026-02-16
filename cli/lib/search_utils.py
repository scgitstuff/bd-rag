import json
import string

# Pylance has problems with this import
# it wants to generate typings, but they add a "porter" module that doesn't seem to exist
# from nltk.stem.porter import PorterStemmer
# both work, so I'm just following the documentation, Pylance can eat all the dicks
from nltk.stem import PorterStemmer  # type: ignore


MOVIES_FILE = "data/movies.json"
STOP_FILE = "data/stopwords.txt"


def loadMovies() -> list[dict[str, str]]:
    with open(MOVIES_FILE, "r") as f:
        data = json.load(f)

    return data["movies"]


def loadStopWords() -> frozenset[str]:
    with open(STOP_FILE, "r") as f:
        lines = f.read().splitlines()

    return frozenset(lines)


def cleanWords(text: str, stopWords: frozenset[str]) -> set[str]:
    text = _preprocess(text)
    words = _tokenize(text)
    words = _removeStopWords(words, stopWords)
    words = _stemWords(words)

    return words


def _stemWords(words: set[str]) -> set[str]:
    out: set[str] = set()
    stemmer = PorterStemmer()

    for word in words:
        out.add(stemmer.stem(word))  # type: ignore

    return out


def _preprocess(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))

    return s


def _tokenize(s: str) -> set[str]:
    words = s.split()
    words = set(words)
    words.discard("")

    return words


def _removeStopWords(words: set[str], stopWords: frozenset[str]) -> set[str]:
    out: set[str] = set()

    for word in words:
        if word not in stopWords:
            out.add(word)

    return out
