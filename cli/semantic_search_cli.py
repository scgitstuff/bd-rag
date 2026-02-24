import argparse
from lib.semantic_search import verifyModel


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subParsers = parser.add_subparsers(dest="command", help="Available commands")

    subParsers.add_parser("verify", help="Verify Semantic Search Model loaded")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verifyModel()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
