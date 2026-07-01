import os



from lib.keyword_search import InvertedIndex
from lib.semantic_search  import ChunkedSemanticSearch


def normalize_scores(scores):
    if len(scores) == 0:
        return None

    min_score = min(scores)
    max_score = max(scores)

    normalized_scores = []
    if min_score == max_score:
        return [1.0]
    
    for score in scores:
        normalized_scores.append((score - min_score) / (max_score - min_score))
    return normalized_scores




class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        if self.idx.load() != None:
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        idx_search = self.idx.bm25_search(query, limit*500)
        sem_search = self.semantic_search.search_chunks(query, limit*500)
        print(idx_search)
        print(sem_search[0])

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")