import argparse
from lib.semantic_search import (
    verifyModel,
    embedTextCommand,
    verifyEmbeddingsCommand,
    embedQueryCommand,
    semanticSearchCommand,
)


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

    embedSearchParser = subParsers.add_parser(
        "search", help="Search for movies using semantic search"
    )
    embedSearchParser.add_argument("query", type=str, help="Search query")
    embedSearchParser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )

    args = parser.parse_args()

    match args.command:
        case "embed_text":
            embedTextCommand(args.text)
        case "verify":
            verifyModel()
        case "verify_embeddings":
            verifyEmbeddingsCommand()
        case "embedquery":
            embedQueryCommand(args.query)
        case "search":
            semanticSearchCommand(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
