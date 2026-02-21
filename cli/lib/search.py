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


def bm25Search(
    movieIndex: InvertedIndex,
    search: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[dict[str, str]]:

    matches: list[dict[str, str]] = []
    searchWords = set(cleanWords(search, movieIndex.stopWords))
    totalScores: dict[int, float] = dict.fromkeys(movieIndex.docmap.keys(), 0.0)

    # score every word
    for word in searchWords:
        docIDs = movieIndex.getDocs(word)

        scores: dict[int, float] = {}
        for docID in docIDs:
            scores[docID] = movieIndex.getBM25(docID, word)
            totalScores[docID] += scores[docID]

    sortedScores = {
        k: v
        for k, v in sorted(
            totalScores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    # add score to output, apply limit
    for docID in sortedScores.keys():
        movie = movieIndex.docmap[docID]
        movie["bm25"] = f"{sortedScores[docID]:.2f}"
        matches.append(movie)

        if len(matches) == limit:
            return matches

    return matches
