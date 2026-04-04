import argparse
import json
import string
import io
from nltk.stem import PorterStemmer

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            main_search(string_format(args.query))
        case _:
            parser.print_help()



def main_search(query):
    movies = string_format(get_movie_data())
    results = token_search(query, movies)

    instance = 1
    for i, movie in enumerate(results, 1):
        print(f"{i}. {movie["title"]}")

    


def token_search(query, movies):
    with open('data/stopwords.txt') as stopdata:
        stopwords = stopdata.read().splitlines()

    stemmer = PorterStemmer()
    tokens = query.split()
    stemmedtokens = []
    for token in tokens:
        if token not in stopwords:
            stemmedtokens.append(stemmer.stem(token))
    tokens = stemmedtokens
    
    results = []
    for token in tokens:
        for movie in movies:
            movietitle = movie["ftitle"]
            if token in movietitle:
                if movietitle not in results:
                    results.append(movie)
    return results




def get_movie_data():
    with open('data/movies.json') as moviedata:
        movies = json.load(moviedata)      
    return movies["movies"]



def string_format(to_format):
    match to_format:
        case str():
            to_format = to_format.lower()
            to_format = to_format.translate(str.maketrans('','', string.punctuation))
            return to_format

        case list():
            return [string_format(item) for item in to_format]
        
        case dict():
            return {**to_format, "ftitle": string_format(to_format["title"])}

        case _:
            print("treating nothing")
            return to_format




if __name__ == "__main__":
    main()

    
