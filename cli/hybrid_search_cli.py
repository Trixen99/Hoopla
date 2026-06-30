import argparse, os, json
from lib.hybrid_search import normalize_scores, HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="convert a range of numbers into their normalized versions")
    normalize_parser.add_argument("scores", nargs='*', type=float, help="numbers to convert")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Search both with keyword & semantic")
    weighted_search_parser.add_argument("query", type=str, help="text to search")
    weighted_search_parser.add_argument("--alpha", nargs='*', default=0.5, type=float, help="constant to dynamically control the weighting between the scores")
    weighted_search_parser.add_argument("--limit", nargs='*', default=5, type=float, help="limit")



    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_numbers(args.scores)

        case "weighted-search":
            weightedSearch(args.query, args.alpha, args.limit)


        case _:
            parser.print_help()


def normalize_numbers(scores):
    nom_scores = normalize_scores(scores)
    if nom_scores == None:
        return
    for score in nom_scores:
        print(f"* {score:.4f}")

def weightedSearch(query, alpha, limit):
    with open(os.path.abspath('data/movies.json')) as moviedata:
        documents = json.load(moviedata)["movies"]  
    hybrid = HybridSearch(documents)
    hybrid.weighted_search(query, alpha, limit)






    
if __name__ == "__main__":
    main()