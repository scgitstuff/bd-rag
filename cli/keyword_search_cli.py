import argparse
from lib.keyword_search import searchKeyWord, buildIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="build the index")

    args = parser.parse_args()

    match args.command:
        case "build":
            buildCommand()
        case "search":
            searchCommand(args.query)
        case _:
            parser.print_help()


def searchCommand(query: str):
    print(f"Searching for: {query}")
    found = searchKeyWord(query)
    for i, movie in enumerate(found, 1):
        print(f"{i}. {movie['title']}")


def buildCommand():

    print("Building inverted index...")

    movieIndex = buildIndex()
    docs = movieIndex.getDocs("merida")
    print(f"First document for token 'merida' = {docs[0]}")

    print("Inverted index built successfully.")


if __name__ == "__main__":
    main()
