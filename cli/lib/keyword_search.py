import string
from .search_utils import loadMovies, loadStopWords


DEFAULT_SEARCH_LIMIT = 5


def searchKeyWord(
    search: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict[str, str]]:

    t = (2, 2, 3)
    _ = t

    movies = loadMovies()
    stopWords = loadStopWords()

    search = preprocess(search)
    searchWords = tokenize(search)
    searchWords = removeStopWords(searchWords, stopWords)

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
