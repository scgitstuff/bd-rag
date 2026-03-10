import json
import regex as re
import string

# Pylance has problems with this import
# it wants to generate typings, but they add a "porter" module that doesn't seem to exist
# from nltk.stem.porter import PorterStemmer
# both work, so I'm just following the documentation, Pylance can eat all the dicks
from nltk.stem import PorterStemmer  # type: ignore


_MOVIES_FILE = "data/movies.json"
_STOP_FILE = "data/stopwords.txt"


def loadMovies() -> list[dict[str, str]]:
    with open(_MOVIES_FILE, "r") as f:
        data = json.load(f)

    return data["movies"]


def loadStopWords() -> frozenset[str]:
    with open(_STOP_FILE, "r") as f:
        lines = f.read().splitlines()

    return frozenset(lines)


# originally I used set()
# that broke Term Frequency, so List it is
def cleanWords(text: str, stopWords: frozenset[str]) -> list[str]:
    out: list[str] = []
    stemmer = PorterStemmer()

    # preprocess
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    # tokenize
    words = text.split()

    for word in words:
        # filter out empty and stop words
        if word and word not in stopWords:
            out.append(stemmer.stem(word))  # type: ignore

    return out


def makeChunks(text: str, chunkSize: int, overlap: int) -> list[str]:
    words = text.split()
    count = len(words)
    start = 0
    end = chunkSize
    chunks: list[str] = []

    while True:
        chunkWords = words[start:end]
        if len(chunkWords) == 0:
            break
        chunks.append(" ".join(chunkWords))

        if end >= count:
            break

        start = end - overlap
        end += chunkSize

    return chunks


def makeSemanticChunks(text: str, maxChunkSize: int, overlap: int) -> list[str]:
    sentences = re.split(pattern=r"(?<=[.!?])\s+", string=text)
    count = len(sentences)
    start = 0
    chunks: list[str] = []

    while True:
        end = start + maxChunkSize

        chunkSentences = sentences[start:end]
        if len(chunkSentences) < 1:
            break
        chunks.append(" ".join(chunkSentences))

        if end >= count:
            break

        start = start + maxChunkSize - overlap

    return chunks
