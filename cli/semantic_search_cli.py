import argparse
from lib.semantic_search import verifyModel, embedText, verifyEmbeddings


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subParsers = parser.add_subparsers(dest="command", help="Available commands")

    subParsers.add_parser("verify", help="Verify that the embedding model is loaded")

    embedParser = subParsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    embedParser.add_argument("text", type=str, help="Text to embed")

    subParsers.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")

    args = parser.parse_args()

    match args.command:
        case "embed_text":
            embedText(args.text)
        case "verify":
            verifyModel()
        case "verify_embeddings":
            verifyEmbeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
