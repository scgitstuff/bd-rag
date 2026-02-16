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


# originally I used set()
# that broke Term Frequency, so List it is
def cleanWords(text: str, stopWords: frozenset[str]) -> list[str]:
    out: list[str] = []
    stemmer = PorterStemmer()

    # preprocess
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    # tokenize
    words = text.split()

    for word in words:
        # filter out empty and stop words
        if word and word not in stopWords:
            out.append(stemmer.stem(word))  # type: ignore

    return out
