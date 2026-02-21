from lib.keyword_search import searchKeyWord, bm25Search
from lib.index import InvertedIndex
from lib.search_utils import loadStopWords


def bm25idfCommand(token: str):
    movieIndex = _loadIndex()
    if movieIndex is None:
        return

    bm25idf = movieIndex.getBM25IDF(token)
    print(f"BM25 IDF score of '{token}': {bm25idf:.2f}")


def bm25tfCommand(id: int, token: str, k1: float, b: float):
    movieIndex = _loadIndex()
    if movieIndex is None:
        return

    score = movieIndex.getBM25TF(id, token, k1, b)
    print(f"BM25 TF score of '{token}' in document '{id}': {score:.2f}")


def buildCommand():
    print("Building inverted index...")

    movieIndex = InvertedIndex(loadStopWords())
    movieIndex.build()
    movieIndex.save()

    print("Inverted index built successfully.")


def idfCommand(token: str):
    movieIndex = _loadIndex()
    if movieIndex is None:
        return

    freq = movieIndex.getIDF(token)
    print(f"Inverse document frequency of '{token}': {freq:.2f}")


def searchCommand(query: str):
    print(f"Searching for: {query}")

    movieIndex = _loadIndex()
    if movieIndex is None:
        return

    movies = searchKeyWord(movieIndex, query)
    for i, movie in enumerate(movies, 1):
        print(f"{i}. {movie['id']} {movie['title']}")


def bm25searchCommand(query: str):
    print(f"Searching for: {query}")

    movieIndex = _loadIndex()
    if movieIndex is None:
        return

    movies = bm25Search(movieIndex, query)
    for i, movie in enumerate(movies, 1):
        print(f"{i}. ({movie['id']}) {movie['title']} - Score: {movie["bm25"]}")


def tfCommand(id: int, token: str):
    movieIndex = _loadIndex()
    if movieIndex is None:
        return

    count = movieIndex.getTF(id, token)
    print(f"'{token}' count in document '{id}' is {count}")


def tfidfCommand(id: int, token: str):
    movieIndex = _loadIndex()
    if movieIndex is None:
        return

    tfidf = movieIndex.getTF_IDF(id, token)
    print(f"TF-IDF score of '{token}' in document '{id}': {tfidf:.2f}")


def _loadIndex() -> InvertedIndex | None:
    movieIndex = InvertedIndex(loadStopWords())
    try:
        movieIndex.load()
    except Exception as e:
        print(e)
        return None

    return movieIndex
