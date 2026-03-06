from sentence_transformers import SentenceTransformer
from typing import Union
from pathlib import Path
import numpy as np
from .search_utils import loadMovies

_EMBED_FILE = Path("cache/movie_embeddings.npy")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents: Union[list[dict[str, str]], None] = None
        self.docmap: dict[int, dict[str, str]] = {}

    def generateEmbedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")

        embeddings = self.model.encode([text])  # type: ignore
        if len(embeddings) < 1:
            raise Exception("encode() returned an empty list")

        return embeddings[0]

    def buildEmbeddings(self, documents: list[dict[str, str]]):
        self.documents = documents
        stuff: list[str] = []

        for movie in documents:
            id = int(movie["id"])
            self.docmap[id] = movie
            stuff.append(f"{movie['title']}: {movie['description']}")

        self.embeddings = self.model.encode(stuff, show_progress_bar=True)  # type: ignore

        np.save(_EMBED_FILE, self.embeddings)

        return self.embeddings

    def loadEmbeddings(self, documents: list[dict[str, str]]):
        if _EMBED_FILE.exists():
            self.embeddings = np.load(_EMBED_FILE)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.buildEmbeddings(documents)


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


def embedText(text: str):
    ss = SemanticSearch()
    embedding = ss.generateEmbedding(text)

    print()
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verifyEmbeddings():
    ss = SemanticSearch()
    movies = loadMovies()
    embeddings = ss.loadEmbeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embedQuery(query: str):
    ss = SemanticSearch()
    embedding = ss.generateEmbedding(query)

    print()
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
