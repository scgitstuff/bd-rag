import argparse
import lib.commands as cmds
import lib.constants as const


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subParsers = parser.add_subparsers(dest="command", help="Available commands")

    bm25idfParser = subParsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25idfParser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25tfParser = subParsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25tfParser.add_argument("docID", type=int, help="Document ID")
    bm25tfParser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25tfParser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=const.BM25_K1,
        help="Tunable BM25 K1 parameter",
    )
    bm25tfParser.add_argument(
        "b",
        type=float,
        nargs="?",
        default=const.BM25_B,
        help="Tunable BM25 b parameter",
    )

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
        case "bm25tf":
            cmds.bm25tfCommand(args.docID, args.term, args.k1, args.b)
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
