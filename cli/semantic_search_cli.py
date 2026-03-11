import argparse
import lib.commands_ss as cmds


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subParsers = parser.add_subparsers(dest="command", help="Available commands")

    subParsers.add_parser("verify", help="Verify that the embedding model is loaded")

    embedParser = subParsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    embedParser.add_argument("text", type=str, help="Text to embed")

    subParsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie dataset"
    )

    embedQueryParser = subParsers.add_parser(
        "embedquery", help="Generate an embedding for a search query"
    )
    embedQueryParser.add_argument("query", type=str, help="Query to embed")

    semanticSearchParser = subParsers.add_parser(
        "search", help="Search for movies using semantic search"
    )
    semanticSearchParser.add_argument("query", type=str, help="Search query")
    semanticSearchParser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )

    chunkParser = subParsers.add_parser(
        "chunk", help="Split text into fixed-size chunks with optional overlap"
    )
    chunkParser.add_argument("text", type=str, help="Text to chunk")
    chunkParser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Size of each chunk in words",
    )
    chunkParser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words to overlap between chunks",
    )

    semanticChunkParser = subParsers.add_parser(
        "semantic_chunk", help="Split text on sentence boundaries to preserve meaning"
    )
    semanticChunkParser.add_argument("text", type=str, help="Text to chunk")
    semanticChunkParser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum size of each chunk in sentences",
    )
    semanticChunkParser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between chunks",
    )

    subParsers.add_parser(
        "embed_chunks", help="Generate embeddings for chunked documents"
    )

    semanticSearchChunkedParser = subParsers.add_parser(
        "search_chunked", help="Search using chunked embeddings"
    )
    semanticSearchChunkedParser.add_argument("query", type=str, help="Search query")
    semanticSearchChunkedParser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )

    args = parser.parse_args()

    match args.command:
        case "embed_text":
            cmds.embedTextCommand(args.text)
        case "verify":
            cmds.verifyModelCommand
        case "verify_embeddings":
            cmds.verifyEmbeddingsCommand()
        case "embedquery":
            cmds.embedQueryCommand(args.query)
        case "search":
            cmds.semanticSearchCommand(args.query, args.limit)
        case "chunk":
            cmds.chunkCommand(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            cmds.semanticChunkCommand(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            cmds.embedChunksCommand()
        case "search_chunked":
            cmds.semanticSearchChunkedCommand(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
