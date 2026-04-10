import json
import string
import io
from nltk.stem import PorterStemmer
import pickle
import os


def main_search(query, index):
    movies = string_format(get_movie_data())
    results = token_search(query, movies)


    instance = 1
    for i, movie in enumerate(results, 1):
        print(f"{i}. {movie["title"]}")

    


def token_search(query, movies):
    with open(get_full_path('data/stopwords.txt')) as stopdata:
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


def get_full_path(path) -> str:
    return f"{os.getcwd()}/{path}"



def get_movie_data():
    with open(get_full_path('data/movies.json')) as moviedata:
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
            return to_format


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
    
    def __add_document(self, doc_id, text):
        tokens = string_format(text.split())
        for token in tokens:
            if token not in self.index:
                self.index[token] = set([doc_id])
            self.index[token].add(doc_id)

    def get_documents(self, term):
        newterm = term.lower()
        if newterm in self.index:
            return sorted(list(self.index[newterm]))

    def build(self):
        moviedata = get_movie_data()
        for movie in moviedata:
            conc = f"{movie["title"]} {movie["description"]}"
            self.__add_document(movie["id"], conc)
            self.docmap[movie["id"]] = movie

    def save(self):
        fullpath = get_full_path("cache/")
        indexpath = f"{fullpath}/index.pkl"
        docmappath = f"{fullpath}/docmap.pkl"

        os.makedirs(fullpath, exist_ok=True)

        with open(indexpath, 'wb') as file:
            pickle.dump(self.index, file)
        with open(docmappath, 'wb') as file:
            pickle.dump(self.docmap, file)


            

        

