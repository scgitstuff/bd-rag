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


def makeFixedChunks(text: str, chunkSize: int, overlap: int) -> list[str]:
    words = text.split()
    count = len(words)
    start = 0
    end = 0
    chunks: list[str] = []

    while True:
        end += chunkSize

        chunkWords = words[start:end]
        if len(chunkWords) > 0:
            chunks.append(" ".join(chunkWords))

        if end >= count:
            break

        start = end - overlap

    return chunks


def makeSemanticChunks(text: str, maxChunkSize: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = list(map(lambda x: str(x), sentences))

    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks: list[str] = []
    i = 0
    sentenceCount = len(sentences)

    while i < sentenceCount:
        chunk_sentences = sentences[i : i + maxChunkSize]
        if chunks and len(chunk_sentences) <= overlap:
            break

        cleaned_sentences: list[str] = []
        for chunk_sentence in chunk_sentences:
            cleaned_sentences.append(chunk_sentence.strip())
        if not cleaned_sentences:
            continue
        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)
        i += maxChunkSize - overlap

    return chunks


# (score - min_score) / (max_score - min_score)
def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    minScore = min(scores)
    maxScore = max(scores)

    if minScore == maxScore:
        return [1.0] * len(scores)

    out: list[float] = []
    for score in scores:
        out.append((score - minScore) / (maxScore - minScore))

    return out
