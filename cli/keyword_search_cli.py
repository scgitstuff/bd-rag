import argparse
import lib.commands as cmds


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subParsers = parser.add_subparsers(dest="command", help="Available commands")

    bm25idfParser = subParsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25idfParser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    subParsers.add_parser("build", help="Build the inverted index")

    idfParser = subParsers.add_parser(
        "idf", help="Get inverse document frequency for a given term"
    )
    idfParser.add_argument("term", type=str, help="Term to get IDF for")

    searchParser = subParsers.add_parser("search", help="Search movies using BM25")
    searchParser.add_argument("query", type=str, help="Search query")

    tfParser = subParsers.add_parser(
        "tf", help="Get term frequency for a given document ID and term"
    )
    tfParser.add_argument("docID", type=int, help="Document ID")
    tfParser.add_argument("term", type=str, help="Term to get frequency for")

    tfidfParser = subParsers.add_parser(
        "tfidf", help="Get TF-IDF score for a given document ID and term"
    )
    tfidfParser.add_argument("docID", type=int, help="Document ID")
    tfidfParser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    args = parser.parse_args()

    match args.command:
        case "bm25idf":
            cmds.bm25idfCommand(args.term)
        case "build":
            cmds.buildCommand()
        case "idf":
            cmds.idfCommand(args.term)
        case "search":
            cmds.searchCommand(args.query)
        case "tf":
            cmds.tfCommand(args.docID, args.term)
        case "tfidf":
            cmds.tfidfCommand(args.docID, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
