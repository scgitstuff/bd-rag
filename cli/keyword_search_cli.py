import argparse
from lib.keyword_search import searchKeyWord
from lib.index import InvertedIndex
from lib.search_utils import loadStopWords


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subParsers = parser.add_subparsers(dest="command", help="Available commands")

    searchParser = subParsers.add_parser("search", help="Search movies using BM25")
    searchParser.add_argument("query", type=str, help="Search query")

    subParsers.add_parser("build", help="build the index")

    tfParser = subParsers.add_parser("tf", help="check count for word in doc")
    tfParser.add_argument("docID", type=int, help="doc ID")
    tfParser.add_argument("token", type=str, help="word to get count for")

    args = parser.parse_args()

    match args.command:
        case "build":
            buildCommand()
        case "search":
            searchCommand(args.query)
        case "tf":
            tfCommand(args.docID, args.token)
        case _:
            parser.print_help()


def searchCommand(query: str):
    print(f"Searching for: {query}")

    movieIndex = InvertedIndex(loadStopWords())
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

    movieIndex = InvertedIndex(loadStopWords())
    movieIndex.build()
    movieIndex.save()

    print("Inverted index built successfully.")


def tfCommand(id: int, token: str):
    movieIndex = InvertedIndex(loadStopWords())
    try:
        movieIndex.load()
    except Exception as e:
        print(e)
        return

    count = movieIndex.getTF(id, token)
    print(f"'{token}' count in document '{id}' is {count}")


if __name__ == "__main__":
    main()
