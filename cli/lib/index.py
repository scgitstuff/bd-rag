from collections import defaultdict
from pathlib import Path
from pickle import dump, load
from .search_utils import cleanWords, loadMovies, loadStopWords


CACHE_DIR = "cache/"
CACHE_INDEX = "index.pkl"
CACHE_MAP = "docmap.pkl"


class InvertedIndex:

    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[str, str]] = {}

    def __addDoc(self, docID: int, words: set[str]):
        for word in words:
            self.index[word].add(docID)

    def getDocs(self, term: str) -> list[int]:
        out = self.index.get(term, set())

        return sorted(out)

    def build(self):
        movies = loadMovies()
        stopWords = loadStopWords()
        
        for m in movies:
            id = int(m["id"])
            text = f"{m['title']} {m['description']}"
            words = cleanWords(text, stopWords)

            self.docmap[id] = m
            self.__addDoc(id, words)

    def save(self):
        cacheDir = Path(CACHE_DIR)
        Path.mkdir(cacheDir, exist_ok=True)

        indexFile = Path(CACHE_DIR + CACHE_INDEX)
        with open(indexFile, "wb") as f:
            dump(self.index, f)

        mapFile = Path(CACHE_DIR + CACHE_MAP)
        with open(mapFile, "wb") as f:
            dump(self.docmap, f)

    def load(self):
        cacheDir = Path(CACHE_DIR)
        mapFile = Path(CACHE_DIR + CACHE_MAP)
        indexFile = Path(CACHE_DIR + CACHE_INDEX)

        filesExist = cacheDir.exists() and mapFile.exists() and indexFile.exists()
        if not filesExist:
            raise FileNotFoundError("cache not found, run build")

        with open(indexFile, "rb") as f:
            self.index = load(f)

        with open(mapFile, "rb") as f:
            self.docmap = load(f)
