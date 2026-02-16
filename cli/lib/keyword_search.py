from .search_utils import cleanWords, loadStopWords
from .index import InvertedIndex


DEFAULT_SEARCH_LIMIT = 5


def searchKeyWord(
    movieIndex: InvertedIndex,
    search: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[dict[str, str]]:

    matches: list[dict[str, str]] = []
    stopWords = loadStopWords()
    searchWords = cleanWords(search, stopWords)

    # I want a full set of matches on all words
    # this will be useful for weights later
    docIDs: set[int] = set()
    for word in searchWords:
        docIDs.update(movieIndex.getDocs(word))
    uniqueIDs = sorted(set(docIDs))

    for id in uniqueIDs:
        movie = movieIndex.docmap[id]
        matches.append(movie)

        if len(matches) == limit:
            break

    return matches


def buildIndex() -> InvertedIndex:
    movieIndex = InvertedIndex()
    movieIndex.build()
    movieIndex.save()

    return movieIndex
