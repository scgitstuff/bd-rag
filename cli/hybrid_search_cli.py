import argparse
import lib.commands_h as cmds


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subParsers = parser.add_subparsers(dest="command", help="Available commands")

    normalizeParser = subParsers.add_parser("normalize", help="normalize")
    normalizeParser.add_argument(
        "scores", nargs="+", type=float, help="List of scores to normalize"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            cmds.normalizeCommand(args.scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
