import argparse
from lib.hybrid_search import normalize_scores

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="convert a range of numbers into their normalized versions")
    normalize_parser.add_argument("scores", nargs='*', type=float, help="numbers to convert")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_numbers(args.scores)

        case _:
            parser.print_help()


def normalize_numbers(scores):
    nom_scores = normalize_scores(scores)


    if nom_scores == None:
        return
    
    for score in nom_scores:
        print(f"* {score:.4f}")

    
if __name__ == "__main__":
    main()