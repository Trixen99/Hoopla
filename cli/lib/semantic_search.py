from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re


def verify_model():
    mod = SemanticSearch()

    print(f"Model loaded: {mod.model}")
    print(f"Max sequence length: {mod.model.max_seq_length}")

def embed_text(text):
    ss_Model = SemanticSearch()
    embedding = ss_Model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss_Model = SemanticSearch()
    with open(os.path.abspath('data/movies.json')) as moviedata:
        documents = json.load(moviedata)["movies"]   
    embeddings = ss_Model.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")




def embed_query_text(query):
    ss_Model = SemanticSearch()
    embed_query = ss_Model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embed_query[:3]}")
    print(f"Shape: {embed_query.shape}")


def search(query, limit):
    ss_Model = SemanticSearch()
    with open(os.path.abspath('data/movies.json')) as moviedata:
        documents = json.load(moviedata)["movies"]   
    embeddings = ss_Model.load_or_create_embeddings(documents)
    results = ss_Model.search(query, limit)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result["title"]} (score: {result["score"]:.4f})\n{result["description"]}\n")





def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def chunk_text(text, size, overlap):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size - overlap)]



def semantic_chunk(text, chunk_size, overlap):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        line = " ".join(sentences[i:i + chunk_size])
        if len(line) > 0:
            chunks.append(line)
        if i + chunk_size >= len(sentences):
            break
    return chunks
  








class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def load_or_create_embeddings(self, documents):
        path = os.path.abspath("cache/embeddings.npy")
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(path) == True:
            self.embeddings = np.load(path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)

        
    def build_embeddings(self, documents):
        self.documents = documents
        movie_list = []
        for document in documents:
            self.document_map[document["id"]] = document
            movie_list.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(movie_list,show_progress_bar=True)
        np.save(os.path.abspath("cache/embeddings.npy"), self.embeddings)
        return self.embeddings




    def generate_embedding(self, text):
        if text == None or len(text) == 0:
            raise ValueError("no text provided")
        return self.model.encode([text])[0]


    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embed = self.generate_embedding(query)
        similarities = []
        for i, key in enumerate(self.document_map):
            tup = (cosine_similarity(self.embeddings[i], query_embed), self.document_map[key])
            similarities.append(tup)
        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
        similarities = similarities[:limit]
        movies = []
        for item in similarities:
            movies.append({"score": item[0], "title": item[1]["title"], "description": item[1]["description"]})
        return movies


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None



    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunks = []
        chunk_metadata = []
        for document in documents:
            self.document_map[document["id"]] = document
            if document["description"] == "" or document["description"] == " ":
                continue
            sem_chunks = semantic_chunk(document["description"], 4, 1)
            for i, chunk in enumerate(sem_chunks):
                chunks.append(chunk)
                chunk_metadata.extend([{"movie_idx": document["id"], "chunk_idx": i, "total_chunks": len(sem_chunks)}])

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save(os.path.abspath("cache/chunk_embeddings.npy"), self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        if os.path.exists(os.path.abspath("cache/chunk_embeddings.npy")) and os.path.exists(os.path.abspath("cache/chunk_metadata.json")):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open(os.path.abspath("cache/chunk_metadata.json"), 'r') as f:
                self.chunk_metadata = json.load(f)
        else:
            self.chunk_embeddings = self.build_chunk_embeddings(documents)
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = 10):
        embedded_query = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk_embed in enumerate(self.chunk_embeddings):
            calculation = (cosine_similarity(self.chunk_embeddings[chunk_embed["chunk_idx"]], embedded_query))
            chunk_scores.append({"chunk_idx": chunk_embed["chunk_idx"], "movie_idx": chunk_embed["movie_idx"], "score": calculation})
        




            tup = (cosine_similarity(self.embeddings[i], query_embed), self.document_map[key])

            chunk_metadata.extend([{"movie_idx": document["id"], "chunk_idx": i, "total_chunks": len(sem_chunks)}])

        

        



