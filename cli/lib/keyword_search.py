import string
from .search_utils import loadMovies  # type: ignore


DEFAULT_SEARCH_LIMIT = 5


def searchKeyWord(
    search: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict[str, str]]:

    movies = loadMovies()

    matches: list[dict[str, str]] = []
    for movie in movies:
        if preprocess(search) in preprocess(movie["title"]):
            matches.append(movie)
            if len(matches) >= limit:
                break

    return matches


def preprocess(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))

    return s
