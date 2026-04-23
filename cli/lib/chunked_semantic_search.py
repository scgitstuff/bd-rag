import json
import numpy as np

from numpy.typing import NDArray
from pathlib import Path
from typing import Any

from .search_utils import makeSemanticChunks
from .semantic_search import SemanticSearch

_CHUNK_EMBED_FILE = Path("cache/chunk_embeddings.npy")
_METADATA_FILE = Path("cache/chunk_metadata.json")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, modelName: str = "all-MiniLM-L6-v2"):
        super().__init__(modelName)

        self.chunkEmbeddings: NDArray[Any] | None = None
        self.chunkMetadata: list[dict[str, int]] | None = None

    def buildChunkEmbeddings(self, documents: list[dict[str, str]]):
        self.documents = documents
        self.docmap = {}
        self.chunkMetadata = []
        allChunks: list[str] = []

        for idx, movie in enumerate(documents):
            text = movie.get("description", "")
            if not text.strip():
                continue

            id = int(movie["id"])
            self.docmap[id] = movie
            movieChunks = makeSemanticChunks(text, maxChunkSize=4, overlap=1)

            chunkCount = len(movieChunks)
            for i, chunk in enumerate(movieChunks):
                allChunks.append(chunk)
                self.chunkMetadata.append(
                    {
                        "movie_idx": idx,
                        "chunk_idx": i,
                        "total_chunks": chunkCount,
                    }
                )

        self.chunkEmbeddings = self.model.encode(allChunks, show_progress_bar=True)  # type: ignore
        np.save(_CHUNK_EMBED_FILE, self.chunkEmbeddings)
        self._saveMetadata(len(allChunks))

        return self.chunkEmbeddings

    def loadChunkEmbeddings(self, documents: list[dict[str, str]]):
        self.documents = documents
        self.document_map = {}

        for movie in documents:
            id = int(movie["id"])
            self.docmap[id] = movie

        if _CHUNK_EMBED_FILE.exists() and _METADATA_FILE.exists():
            self.chunkEmbeddings = np.load(_CHUNK_EMBED_FILE)
            if self.chunkEmbeddings is None:
                return None

            with open(_METADATA_FILE, "r") as f:
                stuff = json.load(f)
                self.chunkMetadata = stuff["chunks"]

            return self.chunkEmbeddings

        return self.buildChunkEmbeddings(documents)

    def searchChunks(self, query: str, limit: int):
        out: list[dict[str, str]] = []

        if self.chunkEmbeddings is None or self.chunkEmbeddings.size == 0:
            print("No chunk embeddings loaded. Call `loadChunkEmbeddings()` first.")
            return out

        # this won't happen, but type checking kept fucking with me
        if self.chunkMetadata is None:
            return out

        queryEmbedding = self.generateEmbedding(query)
        chunkScores: list[dict[str, int | float]] = []

        for i, embedding in enumerate(self.chunkEmbeddings):
            chunkScore = self._cosSimilarity(queryEmbedding, embedding)
            meta = self.chunkMetadata[i]

            chunkScores.append(
                {
                    "chunk_idx": i,
                    "movie_idx": meta["movie_idx"],
                    "score": chunkScore,
                },
            )

        movieScores: dict[int, float] = {}
        for chunkScore in chunkScores:
            movie_idx = int(chunkScore["movie_idx"])
            score = chunkScore["score"]

            if movie_idx not in movieScores or score > movieScores[movie_idx]:
                movieScores[movie_idx] = score

        sortedScores = sorted(movieScores.items(), key=lambda x: x[1], reverse=True)

        if self.documents is None:
            raise ValueError(
                "ChunkedSemanticSearch.documents is None, this should never happen"
            )

        for movie_idx, score in sortedScores[:limit]:
            movie = self.documents[movie_idx]
            movieID = movie["id"]
            title = movie["title"]
            description = movie["description"]
            score = round(score, 4)

            out.append(
                {
                    "id": movieID,
                    "title": title,
                    "description": description,
                    "score": f"{score:.4f}",
                }
            )

        return out

    def _saveMetadata(self, totalChunks: int):
        with open(_METADATA_FILE, "wt") as f:
            json.dump(
                {
                    "chunks": self.chunkMetadata,
                    "total_chunks": totalChunks,
                },
                f,
                indent=2,
            )
