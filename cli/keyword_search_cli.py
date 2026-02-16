import argparse
from lib.keyword_search import buildIndex, searchKeyWord
from lib.index import InvertedIndex


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

    movieIndex = InvertedIndex()
    try:
        movieIndex.load()
    except Exception as e:
        print(e)
        return

    movies = searchKeyWord(movieIndex, query)
    for i, movie in enumerate(movies, 1):
        print(f"{i}. {movie['id']} {movie['title']}")


def buildCommand():
    print("Building inverted index...")
    buildIndex()
    print("Inverted index built successfully.")


if __name__ == "__main__":
    main()
