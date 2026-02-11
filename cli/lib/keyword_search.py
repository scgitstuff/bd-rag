import string
from .search_utils import loadMovies, loadStopWords

# Pylance has problems with this import
# it wants to generate typings, but they add a "porter" module that doesn't seem to exist
# from nltk.stem.porter import PorterStemmer
# both work, so I'm just following the documentation, Pylance can eat all the dicks
from nltk.stem import PorterStemmer  # type: ignore


DEFAULT_SEARCH_LIMIT = 5


def searchKeyWord(
    search: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict[str, str]]:

    movies = loadMovies()
    stopWords = loadStopWords()

    search = preprocess(search)
    searchWords = tokenize(search)
    searchWords = removeStopWords(searchWords, stopWords)
    searchWords = stemWords(searchWords)

    matches: list[dict[str, str]] = []
    for movie in movies:
        title = preprocess(movie["title"])
        title = " ".join(removeStopWords(tokenize(title), stopWords))

        if hasToken(title, searchWords):
            matches.append(movie)
            if len(matches) >= limit:
                break

    return matches


def preprocess(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))

    return s


def tokenize(s: str) -> set[str]:
    words = s.split(" ")
    words = set(words)
    words.discard("")

    return words


def hasToken(title: str, words: set[str]) -> bool:
    for word in words:
        if word in title:
            return True

    return False


def removeStopWords(words: set[str], stopWords: list[str]) -> set[str]:
    out: set[str] = set()
    for word in words:
        if word not in stopWords:
            out.add(word)

    return out


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
