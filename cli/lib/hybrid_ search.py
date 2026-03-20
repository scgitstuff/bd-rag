from .index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .search_utils import loadStopWords
from .keyword_search import bm25Search


class HybridSearch:
    def __init__(self, documents: list[dict[str, str]]):
        self.documents = documents
        self.search = ChunkedSemanticSearch()
        self.search.loadChunkEmbeddings(documents)
        self.idx = InvertedIndex(loadStopWords())
        self.idx.load()

    def _bm25Search(self, query: str, limit: int):
        return bm25Search(self.idx, query, limit)

    def weightedSearch(self, query: str, alpha: float, limit: int) -> None:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrfSearch(self, query: str, k: float, limit: int) -> None:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
