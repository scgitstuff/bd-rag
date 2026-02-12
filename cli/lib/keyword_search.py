from .search_utils import loadMovies, loadStopWords, cleanWords, stemWords
from .index import InvertedIndex


DEFAULT_SEARCH_LIMIT = 5


def searchKeyWord(
    search: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[dict[str, str]]:

    movies = loadMovies()
    stopWords = loadStopWords()

    searchWords = cleanWords(search, stopWords)
    searchWords = stemWords(searchWords)

    matches: list[dict[str, str]] = []
    for movie in movies:
        title = " ".join(cleanWords(movie["title"]))

        if _hasToken(title, searchWords):
            matches.append(movie)
            if len(matches) >= limit:
                break

    return matches


def _hasToken(title: str, words: set[str]) -> bool:
    for word in words:
        if word in title:
            return True

    return False


def buildIndex() -> InvertedIndex:
    movies = loadMovies()

    movieIndex = InvertedIndex()
    movieIndex.build(movies)
    movieIndex.save()

    return movieIndex
