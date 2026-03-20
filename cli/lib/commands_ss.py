from .search_utils import loadMovies, makeFixedChunks, makeSemanticChunks
from .semantic_search import SemanticSearch
from .chunked_semantic_search import ChunkedSemanticSearch


def verifyModelCommand():
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
    stuff = ss.search(query, limit)

    for i, thingy in enumerate(stuff, 1):
        print("\n\n")
        print(f"{i}. {thingy["title"]} (score: {thingy["score"]:.4f})")
        print(f"   {thingy["description"]}")


def chunkCommand(text: str, chunk_size: int, overlap: int):
    chunks = makeFixedChunks(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def semanticChunkCommand(text: str, maxChunkSize: int, overlap: int):
    chunks = makeSemanticChunks(text, maxChunkSize, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def embedChunksCommand():
    css = ChunkedSemanticSearch()
    movies = loadMovies()
    embeddings = css.loadChunkEmbeddings(movies)

    if embeddings is None:
        print(f"embeddings is None")
        return

    print(f"Generated {len(embeddings)} chunked embeddings")


def semanticSearchChunkedCommand(query: str, limit: int):
    css = ChunkedSemanticSearch()
    movies = loadMovies()
    css.loadChunkEmbeddings(movies)
    results = css.searchChunks(query, limit)

    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res["title"]} (score: {res["score"]})")
        print(f"   {res["document"]}...")
