from .index import InvertedIndex
from .search_utils import loadStopWords


def loadIndex() -> InvertedIndex | None:
    movieIndex = InvertedIndex(loadStopWords())
    try:
        movieIndex.load()
    except Exception as e:
        print(e)
        return None

    return movieIndex
