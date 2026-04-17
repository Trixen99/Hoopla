import argparse
from lib.keyword_search import main_search
from lib.keyword_search import string_format
from lib.keyword_search import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build Index and Cache files")

    search_parser = subparsers.add_parser("tf", help="Prints the term frequency for the given term in the document with the given ID")
    search_parser.add_argument("movie_id", type=str, help="Movie ID to Look through")
    search_parser.add_argument("term", type=str, help="Term to Search in Movie")


    args = parser.parse_args()


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            main_search(string_format(args.query))
        
        case "build":
            build()

        case "tf":
            termcount(int(args.movie_id), args.term)


        case _:
            parser.print_help()

def build():
    index = InvertedIndex()
    index.build()
    index.save()
    return index


def termcount(movie_id, term):
    print(f"Looking for {term} in Movie Number {movie_id}" )
    index = InvertedIndex()
    index.load()
    print(index.get_tf(movie_id, term))

if __name__ == "__main__":
    main()

    
