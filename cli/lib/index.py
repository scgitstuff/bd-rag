from collections import Counter, defaultdict
import math
from pathlib import Path
from pickle import dump, load
from functools import reduce
from .search_utils import cleanWords, loadMovies
import lib.constants as const


_CACHE_DIR = "cache/"
_INDEX_FILE = Path(_CACHE_DIR + "index.pkl")
_MAP_FILE = Path(_CACHE_DIR + "docmap.pkl")
_FREQUENCY_FILE = Path(_CACHE_DIR + "term_frequencies.pkl")
_DOC_LEN_FILE = Path(_CACHE_DIR + "doc_lengths.pkl")


class InvertedIndex:

    def __init__(self, stopWords: frozenset[str]):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[str, str]] = {}
        self.termFrequency: dict[int, Counter[str]] = defaultdict(Counter)
        self.docLengths: dict[int, int] = {}
        self.stopWords = stopWords

    def getDocs(self, term: str) -> list[int]:
        out = self.index.get(term, set())
        return sorted(list(out))

    def getIDF(self, term: str) -> float:
        term = self.__getSingleToken(term)
        docCount = len(self.docmap)
        termDocCount = len(self.getDocs(term))
        # print(f"math.log(({docCount} + 1) / ({termDocCount} + 1))")

        return math.log((docCount + 1) / (termDocCount + 1))

    def getTF(self, docID: int, term: str) -> int:
        term = self.__getSingleToken(term)

        return self.termFrequency[docID][term]

    def getTF_IDF(self, docID: int, term: str) -> float:
        tf = self.getTF(docID, term)
        idf = self.getIDF(term)

        return tf * idf

    def getBM25IDF(self, term: str) -> float:
        term = self.__getSingleToken(term)
        docCount = len(self.docmap)
        termDocCount = len(self.getDocs(term))
        idfBM25 = math.log((docCount - termDocCount + 0.5) / (termDocCount + 0.5) + 1)

        return idfBM25

    def getBM25TF(
        self,
        docID: int,
        term: str,
        k1: float = const.BM25_K1,
        b: float = const.BM25_B,
    ) -> float:
        tf = self.getTF(docID, term)

        # Length normalization factor
        normalizedLen = 1 - b + b * (self.docLengths[docID] / self.__avgDocLen())

        # Apply normalize to term frequency
        saturation = (tf * (k1 + 1)) / (tf + k1 * normalizedLen)

        return saturation

    def build(self):
        movies = loadMovies()

        for m in movies:
            id = int(m["id"])
            text = f"{m['title']} {m['description']}"

            self.docmap[id] = m
            self.__addDoc(id, text)

    def save(self):
        Path.mkdir(Path(_CACHE_DIR), exist_ok=True)

        with open(_INDEX_FILE, "wb") as f:
            dump(self.index, f)

        with open(_MAP_FILE, "wb") as f:
            dump(self.docmap, f)

        with open(_FREQUENCY_FILE, "wb") as f:
            dump(self.termFrequency, f)

        with open(_DOC_LEN_FILE, "wb") as f:
            dump(self.docLengths, f)

    def load(self):
        filesExist = (
            Path(_CACHE_DIR).exists()
            and _INDEX_FILE.exists()
            and _MAP_FILE.exists()
            and _FREQUENCY_FILE.exists()
            and _DOC_LEN_FILE.exists()
        )
        if not filesExist:
            raise FileNotFoundError("cache not found, run build")

        with open(_INDEX_FILE, "rb") as f:
            self.index = load(f)

        with open(_MAP_FILE, "rb") as f:
            self.docmap = load(f)

        with open(_FREQUENCY_FILE, "rb") as f:
            self.termFrequency = load(f)

        with open(_DOC_LEN_FILE, "rb") as f:
            self.docLengths = load(f)

    def __addDoc(self, docID: int, text: str):
        words = cleanWords(text, self.stopWords)
        for word in set(words):
            self.index[word].add(docID)

        self.termFrequency[docID].update(words)
        self.docLengths[docID] = len(words)

    def __getSingleToken(self, term: str) -> str:
        # I used set to alow for duplicate stems
        stuff = set(cleanWords(term, self.stopWords))
        if len(stuff) != 1:
            raise ValueError("term must be a single token")
        term = stuff.pop()

        return term

    def __avgDocLen(self) -> float:
        count = len(self.docLengths.keys())

        if count < 1:
            return 0.0

        sum_ = reduce(lambda a, b: a + b, self.docLengths.values())

        return sum_ / count
