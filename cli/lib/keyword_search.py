import string
from .search_utils import loadMovies  # type: ignore


DEFAULT_SEARCH_LIMIT = 5


def searchKeyWord(
    search: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict[str, str]]:
    movies = loadMovies()
    search = preprocess(search)
    searchWords = tokenize(search)

    matches: list[dict[str, str]] = []
    for movie in movies:
        title = preprocess(movie["title"])

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


def hasToken(title: str, searchWords: set[str]) -> bool:
    for word in searchWords:
        if word in title:
            return True

    return False
