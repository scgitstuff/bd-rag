from .search_utils import normalize
from .hybrid_search import HybridSearch
from .commands_util import loadIndex
from .search_utils import loadMovies
from .enhance import enhanceQuery


def normalizeCommand(scores: list[float]):
    print(scores)

    scores = normalize(scores)

    for score in scores:
        print(f"* {score:.4f}")


def weightedSearchCommand(query: str, alpha: float, limit: int):
    movieIndex = loadIndex()
    if movieIndex is None:
        return

    movies = loadMovies()
    hs = HybridSearch(movies, movieIndex)
    movies = hs.weightedSearch(query, alpha, limit)

    print()
    print()
    print(f"Weighted Hybrid Search Results for '{query}' (alpha={alpha}):")
    print(
        f"  Alpha {alpha}: {int(alpha * 100)}% Keyword, {int((1 - alpha) * 100)}% Semantic"
    )

    for i, movie in enumerate(movies, 1):
        hybrid = float(movie["hybrid"])
        bm25 = float(movie["bm25"])
        semantic = float(movie["semantic"])

        print()
        print(f"{i}. {movie["title"]}")
        print(f"   Hybrid Score: {hybrid:.3f}")
        print(f"   BM25: {bm25:.3f}, Semantic: {semantic:.3f}")
        print(f"   {movie['description'][:100]}")


def rrfSearchCommand(query: str, k: int, limit: int, enhance: str, rerankMethod: str):
    movieIndex = loadIndex()
    if movieIndex is None:
        return

    query = enhanceQuery(enhance, query)

    movies = loadMovies()
    hs = HybridSearch(movies, movieIndex)
    movies = hs.rrfSearch(query, k, limit)

    for i, movie in enumerate(movies, 1):
        rrf = float(movie["rrf"])
        bm25Rank = int(movie["bm25Rank"])
        semanticRank = int(movie["semanticRank"])

        print()
        print(f"{i}. {movie["title"]}")
        print(f"   RRF Score: {rrf:.3f}")
        print(f"   BM25 Rank: {bm25Rank}, Semantic Rank: {semanticRank}")
        print(f"   {movie['description'][:100]}")
