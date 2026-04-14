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

    args = parser.parse_args()


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            main_search(string_format(args.query))
        
        case "build":
            index = build()

        case _:
            parser.print_help()

def build():
    index = InvertedIndex()
    index.build()
    index.save()
    return index

if __name__ == "__main__":
    main()

    
