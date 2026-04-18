import argparse
import lib.commands_h as cmds


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subParsers = parser.add_subparsers(dest="command", help="Available commands")

    normalizeParser = subParsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalizeParser.add_argument(
        "scores", nargs="+", type=float, help="List of scores to normalize"
    )

    weightedSearchParser = subParsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weightedSearchParser.add_argument("query", type=str, help="Search query")
    weightedSearchParser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
    )
    weightedSearchParser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    rrfSearchParser = subParsers.add_parser(
        "rrf-search", help="Perform Reciprocal Rank Fusion search"
    )
    rrfSearchParser.add_argument("query", type=str, help="Search query")
    rrfSearchParser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF k parameter controlling weight distribution (default=60)",
    )
    rrfSearchParser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )
    rrfSearchParser.add_argument(
        "--enhance",
        type=str,
        choices=["spell"],
        help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            cmds.normalizeCommand(args.scores)
        case "weighted-search":
            cmds.weightedSearchCommand(args.query, args.alpha, args.limit)
        case "rrf-search":
            cmds.rrfSearchCommand(args.query, args.k, args.limit, args.enhance)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
