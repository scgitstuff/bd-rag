from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path
import numpy as np

from numpy.typing import NDArray
from .search_utils import loadMovies

_EMBED_FILE = Path("cache/movie_embeddings.npy")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # this should be the typing, but does not work
        # I'm not familiar enough with numpy to make it work
        # self.embeddings: Union[NDArray[NDArray[np.float32]], None] = None
        self.embeddings: Union[NDArray[Any], None] = None
        self.documents: Union[list[dict[str, str]], None] = None
        self.docmap: dict[int, dict[str, str]] = {}

    def generateEmbedding(self, text: str) -> NDArray[np.float32]:
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")

        embeddings = self.model.encode([text])  # type: ignore

        if len(embeddings) < 1:
            raise Exception("encode() returned an empty list")

        # print("generateEmbedding()")
        # print(type(embeddings))
        # print(type(embeddings[0]))  # type: ignore
        # print(type(embeddings[0][0]))  # type: ignore

        return embeddings[0]

    def buildEmbeddings(self, documents: list[dict[str, str]]):
        self.documents = documents
        self.document_map = {}
        movieStrings: list[str] = []

        for movie in documents:
            id = int(movie["id"])
            self.docmap[id] = movie
            movieStrings.append(f"{movie['title']}: {movie['description']}")

        self.embeddings = self.model.encode(movieStrings, show_progress_bar=True)  # type: ignore
        np.save(_EMBED_FILE, self.embeddings)

        return self.embeddings

    def loadEmbeddings(self, documents: list[dict[str, str]]):
        self.documents = documents
        self.document_map = {}

        for movie in documents:
            id = int(movie["id"])
            self.docmap[id] = movie

        if _EMBED_FILE.exists():
            self.embeddings = np.load(_EMBED_FILE)
            if self.embeddings is None:
                return None
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.buildEmbeddings(documents)

    def search(self, query: str, limit: int):
        out: List[Dict[str, str | float]] = []

        if self.embeddings is None or self.embeddings.size == 0:
            print("No embeddings loaded. Call `loadEmbeddings()` first.")
            return out

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        queryEmbedding = self.generateEmbedding(query)
        similarities: List[Tuple[float, Dict[str, str]]] = []

        for i, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(queryEmbedding, embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        for score, doc in similarities[:limit]:
            out.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return out


def verifyModel():
    # import torch
    # if torch.cuda.is_available():
    #     print(f"\nDEVICE: {torch.cuda.get_device_name(0)}")
    #     print(torch.__version__)
    #     print(torch.version.cuda)  # type: ignore
    #     print(torch.cuda.get_arch_list())
    # else:
    #     print("\nCUDA is not available")

    ss = SemanticSearch()
    print()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embedTextCommand(text: str):
    ss = SemanticSearch()
    embedding = ss.generateEmbedding(text)

    print()
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verifyEmbeddingsCommand():
    ss = SemanticSearch()
    movies = loadMovies()
    embeddings = ss.loadEmbeddings(movies)
    if embeddings is None:
        print(f"embeddings is None")
        return

    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embedQueryCommand(query: str):
    ss = SemanticSearch()
    embedding = ss.generateEmbedding(query)

    print()
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def semanticSearchCommand(query: str, limit: int):
    ss = SemanticSearch()
    movies = loadMovies()
    ss.loadEmbeddings(movies)

    # print(f"\nLIMIT: {limit}\n")

    stuff = ss.search(query, limit)

    for i, thingy in enumerate(stuff, 1):
        print("\n\n")
        print(f"{i}. {thingy["title"]} (score: {thingy["score"]:.4f})")
        print(f"   {thingy["description"]}")


# code provided by lesson
def cosine_similarity(vec1: NDArray[np.float32], vec2: NDArray[np.float32]) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
