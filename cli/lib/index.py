from pathlib import Path
from pickle import dump
from .search_utils import cleanWords
from collections import defaultdict

CACHE_DIR = "cache/"
CACHE_INDEX = "index.pkl"
CACHE_MAP = "docmap.pkl"


class InvertedIndex:

    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, str] = {}

    def __addDoc(self, docID: int, text: str):
        self.docmap[docID] = text
        words = cleanWords(text)

        for word in words:
            self.index[word].add(docID)

    def getDocs(self, term: str) -> list[int]:
        out = self.index.get(term, set())

        return sorted(out)

    def build(self, movies: list[dict[str, str]]):
        for m in movies:
            id = int(m["id"])
            text = f"{m['title']} {m['description']}"
            self.__addDoc(id, text)

    def save(self):
        cacheDir = Path(CACHE_DIR)
        Path.mkdir(cacheDir, exist_ok=True)

        indexFile = Path(CACHE_DIR + CACHE_INDEX)
        with open(indexFile, "wb") as f:
            dump(self.index, f)

        mapFile = Path(CACHE_DIR + CACHE_MAP)
        with open(mapFile, "wb") as f:
            dump(self.docmap, f)
