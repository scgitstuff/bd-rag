import argparse
import lib.commands as cmds


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subParsers = parser.add_subparsers(dest="command", help="Available commands")

    searchParser = subParsers.add_parser("search", help="Search movies using BM25")
    searchParser.add_argument("query", type=str, help="Search query")

    subParsers.add_parser("build", help="build the index")

    tfParser = subParsers.add_parser("tf", help="check count for word in doc")
    tfParser.add_argument("docID", type=int, help="doc ID")
    tfParser.add_argument("token", type=str, help="word to get count for")

    idfParser = subParsers.add_parser("idf", help="Inverse Document Frequency")
    idfParser.add_argument("token", type=str, help="word to get frequency for")

    args = parser.parse_args()

    match args.command:
        case "build":
            cmds.buildCommand()
        case "idf":
            cmds.idfCommand(args.token)
        case "search":
            cmds.searchCommand(args.query)
        case "tf":
            cmds.tfCommand(args.docID, args.token)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
