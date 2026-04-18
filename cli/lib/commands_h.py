from .search_utils import normalize
from .hybrid_search import HybridSearch
from .commands_util import loadIndex
from .search_utils import loadMovies
import os
from dotenv import load_dotenv
from google import genai


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


def rrfSearchCommand(query: str, k: int, limit: int, enhance: str):
    movieIndex = loadIndex()
    if movieIndex is None:
        return

    if enhance == "spell":
        query = _spellCheck(query)

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


# the solution runs this code in rrfSearch()
# I think the solution is doing it wrong
# spell check is a pre-process on the query, not part of the search functionality
def _spellCheck(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    contents = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    response = client.models.generate_content(  # type: ignore
        model="gemma-3-27b-it",
        contents=contents,
    )

    out: str = ""
    if response.text:
        out = response.text

    # print(f"\nSpell Check Before: \n{query}")
    # print(f"\nSpell Check After: \n{out}\n")
    print(f"\nEnhanced query (spell): '{query}' -> '{out}'\n")

    return out
