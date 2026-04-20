import json
import string
import io
from nltk.stem import PorterStemmer
import pickle
import os
from collections import Counter 
import math


stopwords = None

def main_search(query):
    index = InvertedIndex()
    status = index.load()
    if status is not None:
        print(status)
        return

    print(index.term_frequencies)
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
        self.term_frequencies = {}
    
    def __add_document(self, doc_id, text):
        tokens = tokenize(text)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            self.term_frequencies[doc_id].update([token])
            if token not in self.index:
                self.index[token] = set([doc_id])
            else:
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
        freq_path = f"{fullpath}/term_frequencies.pkl"

        os.makedirs(fullpath, exist_ok=True)

        with open(indexpath, 'wb') as file:
            pickle.dump(self.index, file)
        with open(docmappath, 'wb') as file:
            pickle.dump(self.docmap, file)
        with open(freq_path, 'wb') as file:
            pickle.dump(self.term_frequencies, file)


    def load(self) -> str:
        with open (get_full_path("cache/index.pkl"), "rb") as file_index:
            self.index = pickle.load(file_index)
        with open (get_full_path("cache/docmap.pkl"), "rb") as file_docmap:
            self.docmap = pickle.load(file_docmap)
        with open (get_full_path("cache/term_frequencies.pkl"), "rb") as file_freq:
            self.term_frequencies = pickle.load(file_freq)

        if self.index == {} or self.term_frequencies == {} or self.docmap == {}:
            return "one or more pre-built files missing, please re-build."
        return None

    def get_tf(self, doc_id, term) -> int:
        term_list = tokenize(term)
        if len(term_list) > 1:
            raise Exception("More than one token returned when retrieving term count")

        token_term = term_list[0]
        
        if doc_id not in self.term_frequencies:
            return 0 

        if token_term not in self.term_frequencies[doc_id]:
            return 0

        return self.term_frequencies[doc_id][token_term]

    def get_bm25_idf(self, term) -> float:
        tokens = tokenize(term)

        total = 0
        for token in tokens:
            instances = len(self.get_documents(token))
            total += math.log((len(self.docmap) - instances + 0.5) / (instances + 0.5) + 1)
        return total





            

        

