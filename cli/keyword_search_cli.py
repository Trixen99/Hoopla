import argparse
from lib.keyword_search import main_search
from lib.keyword_search import string_format
from lib.keyword_search import InvertedIndex
from lib.keyword_search import tokenize
import math

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Index and Cache files")

    tf_parser = subparsers.add_parser("tf", help="Prints the term frequency for the given term in the document with the given ID")
    tf_parser.add_argument("movie_id", type=str, help="Movie ID to Look through")
    tf_parser.add_argument("term", type=str, help="Term to Search in Movie")


    idf_parser = subparsers.add_parser("idf", help="Calculate the IDF of a term")
    idf_parser.add_argument("term", type=str, help="Term to calculate")


    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate the TF-IDF score for term")
    tfidf_parser.add_argument("movie_id", type=str, help="Movie ID to calculate")
    tfidf_parser.add_argument("term", type=str, help="Term to calculate")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")


    args = parser.parse_args()


    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            main_search(string_format(args.query))
        
        case "build":
            build()

        case "tf":
            termcount(int(args.movie_id), args.term)

        case "idf":
            calclog(args.term)

        case "tfidf":
            calc_tfidf(int(args.movie_id), args.term)

        case "bm25idf":
            bm25_idf_command(args.term)

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



def calclog(term):
    index = InvertedIndex()
    index.load()
    token_list = tokenize(term)
    if len(token_list) > 1:
        raise Exception("More than one token returned when retrieving term count")
    token = token_list[0]

    log = math.log((len(index.docmap) + 1) / (len(index.get_documents(tokenize(token)[0]))+ 1))

    print(f"Inverse document frequency of '{term}': {log:.2f}")
    
def calc_tfidf(movie_id, term):
    index = InvertedIndex()
    index.load()
    token_list = tokenize(term)

    total_tdidf = 0

    for token in token_list:
        idf = math.log((len(index.docmap) + 1) / (len(index.get_documents(token))+ 1))
        tf = index.get_tf(movie_id, token)
        total_tdidf += (idf * tf)

    print(f"TF-IDF score of '{term}' in document '{movie_id}': {total_tdidf:.2f}")



def bm25_idf_command(term):
    index = InvertedIndex()
    index.load()
    score = index.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {score:.2f}")

        

if __name__ == "__main__":
    main()

    
