import json
import string
import io
from nltk.stem import PorterStemmer
import pickle
import os


stopwords = None

def main_search(query):
    index = InvertedIndex()
    status = index.load()
    if status is not None:
        print(status)
        return
    tokens = tokenize(query)


    docs = []
    for token in tokens:
        documents = index.get_documents(token)
        if documents is not None:
            docs.extend(documents)
    for i, doc in enumerate(docs):
        if i > 4:
            break
        print(index.docmap[doc]["title"])


def get_full_path(path) -> str:
    return f"{os.getcwd()}/{path}"


def get_movie_data():
    with open(get_full_path('data/movies.json')) as moviedata:
        movies = json.load(moviedata)      
    return movies["movies"]

def get_stop_words():
    global stopwords
    if stopwords is None:
        with open(get_full_path('data/stopwords.txt')) as stopdata:
            stopwords = stopdata.read().splitlines()
    return stopwords
        


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

def tokenize(to_format):
    stemmer = PorterStemmer()
    tokens = string_format(to_format.split())
    stemmedtokens = []

    for token in tokens:
        if token not in get_stop_words():
            stemmedtokens.append(stemmer.stem(token))
    tokens = stemmedtokens
    return tokens




class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
    
    def __add_document(self, doc_id, text):
        tokens = tokenize(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set([doc_id])
            self.index[token].add(doc_id)

    def get_documents(self, term):
        if term in self.index:
            return sorted(list(self.index[term]))

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

    def load(self) -> str:
        with open (get_full_path("cache/index.pkl"), "rb") as file_index:
            self.index = pickle.load(file_index)
        with open (get_full_path("cache/docmap.pkl"), "rb") as file_docmap:
            self.docmap = pickle.load(file_docmap)

        if self.docmap == {} and self.index == {}:
            return "no pre-built index or docmap file found"

        if self.index == {}:
            return "no pre-built index file found"
        
        if self.docmap == {}:
            return "no pre-built docmap file found"

        return None

            

        

