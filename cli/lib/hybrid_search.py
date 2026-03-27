from .index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .search_utils import normalize
from .keyword_search import bm25Search


class HybridSearch:
    def __init__(self, movies: list[dict[str, str]], idx: InvertedIndex):
        self.movies = movies
        self.idx = idx

        self.search = ChunkedSemanticSearch()
        self.search.loadChunkEmbeddings(movies)

    def _bm25Search(self, query: str, limit: int):
        return bm25Search(self.idx, query, limit)

    def weightedSearch(
        self, query: str, alpha: float, limit: int
    ) -> list[dict[str, str]]:
        bigLimit = limit * 500

        # search
        bm25Results = self._bm25Search(query, bigLimit)
        semanticResults = self.search.searchChunks(query, bigLimit)

        # normalize scores
        bm25Scores: list[float] = list(map(lambda x: float(x["bm25"]), bm25Results))
        semanticScores: list[float] = list(
            map(lambda x: float(x["score"]), semanticResults)
        )
        bm25NormScores = normalize(bm25Scores)
        semanticNormScores = normalize(semanticScores)

        # build dict union of both searches
        allScores: dict[int, dict[str, str]] = {}
        for i in range(len(bm25Results)):
            id = int(bm25Results[i]["id"])
            bm25 = bm25NormScores[i]
            allScores[id] = allScores.get(id, {})
            allScores[id]["bm25"] = f"{bm25}"
        for i in range(len(semanticResults)):
            id = int(semanticResults[i]["id"])
            semantic = semanticNormScores[i]
            allScores[id] = allScores.get(id, {})
            allScores[id]["semantic"] = f"{semantic}"

        # calculate hybrid score
        for id, d in allScores.items():
            # fill in blanks for IDs that only existed in one list
            d["bm25"] = d.get("bm25", "0")
            d["semantic"] = d.get("semantic", "0")

            hybrid = _hybridScore(float(d["bm25"]), float(d["semantic"]), alpha)
            d["hybrid"] = f"{hybrid}"

        # sort on hybrid
        sortedScores: dict[int, dict[str, str]] = {
            id: d
            for id, d in sorted(
                allScores.items(),
                key=lambda item: float(item[1]["hybrid"]),
                reverse=True,
            )
        }

        # apply limit, add details for top results
        out: list[dict[str, str]] = []
        for id, d in sortedScores.items():
            d["id"] = f"{id}"
            d["title"] = self.idx.docmap[id]["title"]
            d["description"] = self.idx.docmap[id]["description"]

            out.append(d)
            if len(out) == limit:
                return out

        return out

    def rrfSearch(self, query: str, k: float, limit: int) -> None:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def _hybridScore(bm25Score: float, semanticScore: float, alpha: float = 0.5) -> float:
    return alpha * bm25Score + (1 - alpha) * semanticScore
