from collections import Counter, defaultdict
from pathlib import Path
from pickle import dump, load
from .search_utils import cleanWords, loadMovies


CACHE_DIR = "cache/"
CACHE_INDEX = "index.pkl"
CACHE_MAP = "docmap.pkl"
CACHE_FREQ = "term_frequencies.pkl"


class InvertedIndex:

    def __init__(self, stopWords: frozenset[str]):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[str, str]] = {}
        self.termFrequency: dict[int, Counter[str]] = defaultdict(Counter)
        self.stopWords = stopWords

    def __addDoc(self, docID: int, text: str):
        words = cleanWords(text, self.stopWords)
        for word in set(words):
            self.index[word].add(docID)

        self.termFrequency[docID].update(words)

    def getDocs(self, term: str) -> list[int]:
        out = self.index.get(term, set())
        return sorted(list(out))

    def getTF(self, docID: int, term: str) -> int:
        # I used set to alow for duplicate stems
        stuff = set(cleanWords(term, self.stopWords))
        if len(stuff) != 1:
            raise Exception("there can be only one! token")

        # should we tell user what the stem is?
        s = stuff.pop()
        # print(f"{term} -> {s}")

        return self.termFrequency[docID][s]

    def build(self):
        movies = loadMovies()

        for m in movies:
            id = int(m["id"])
            text = f"{m['title']} {m['description']}"

            self.docmap[id] = m
            self.__addDoc(id, text)

    def save(self):
        cacheDir = Path(CACHE_DIR)
        indexFile = Path(CACHE_DIR + CACHE_INDEX)
        mapFile = Path(CACHE_DIR + CACHE_MAP)
        freqFile = Path(CACHE_DIR + CACHE_FREQ)

        Path.mkdir(cacheDir, exist_ok=True)

        with open(indexFile, "wb") as f:
            dump(self.index, f)

        with open(mapFile, "wb") as f:
            dump(self.docmap, f)

        with open(freqFile, "wb") as f:
            dump(self.termFrequency, f)

    def load(self):
        cacheDir = Path(CACHE_DIR)
        indexFile = Path(CACHE_DIR + CACHE_INDEX)
        mapFile = Path(CACHE_DIR + CACHE_MAP)
        freqFile = Path(CACHE_DIR + CACHE_FREQ)

        filesExist = (
            cacheDir.exists()
            and mapFile.exists()
            and indexFile.exists()
            and freqFile.exists()
        )
        if not filesExist:
            raise FileNotFoundError("cache not found, run build")

        with open(indexFile, "rb") as f:
            self.index = load(f)

        with open(mapFile, "rb") as f:
            self.docmap = load(f)

        with open(freqFile, "rb") as f:
            self.termFrequency = load(f)
