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


def loadStopWords() -> list[str]:
    with open(STOP_FILE, "r") as f:
        return f.read().splitlines()


def cleanWords(text: str, stopWords: list[str] | None = None) -> set[str]:
    if stopWords is None:
        stopWords = []

    text = _preprocess(text)
    words = _tokenize(text)
    if len(stopWords) > 0:
        words = _removeStopWords(words, stopWords)

    return words


def stemWords(words: set[str]) -> set[str]:
    out: set[str] = set()
    stemmer = PorterStemmer()

    out = set(
        map(
            lambda word: stemmer.stem(word),  # type: ignore
            words,
        )
    )

    return out


def _preprocess(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))

    return s


def _tokenize(s: str) -> set[str]:
    words = s.split(" ")
    words = set(words)
    words.discard("")

    return words


def _removeStopWords(words: set[str], stopWords: list[str]) -> set[str]:
    out: set[str] = set()

    for word in words:
        if word not in stopWords:
            out.add(word)

    return out
