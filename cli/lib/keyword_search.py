from .search_utils import cleanWords
from .index import InvertedIndex


DEFAULT_SEARCH_LIMIT = 5


def searchKeyWord(
    movieIndex: InvertedIndex,
    search: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[dict[str, str]]:

    matches: list[dict[str, str]] = []
    seen: set[int] = set()
    searchWords = set(cleanWords(search, movieIndex.stopWords))

    for word in searchWords:
        docIDs = movieIndex.getDocs(word)
        for id in docIDs:
            if id in seen:
                continue
            seen.add(id)

            movie = movieIndex.docmap[id]
            matches.append(movie)

            if len(matches) == limit:
                return matches

    return matches
