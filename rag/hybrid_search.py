from rank_bm25 import BM25Okapi
import numpy as np


class HybridSearcher:
    def __init__(self, chunks: list[dict], collection, alpha=0.5):
        '''
        chunks: list of {'text': str, 'page': int}
        collection: ChromaDB collection
        alpha: weight for vector search (0=only BM25, 1=only vector)
        '''
        self.chunks = chunks
        self.collection = collection
        self.alpha = alpha

        tokenized = [c['text'].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, n_results=20) -> list[dict]:
        # ── BM25 scores ──
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # ── Vector scores ──
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, len(self.chunks))
        )
        vector_ids = vector_results['ids'][0]
        vector_distances = vector_results['distances'][0]

        vector_scores = np.zeros(len(self.chunks))
        for chunk_id, dist in zip(vector_ids, vector_distances):
            idx = int(chunk_id.split('_')[1])
            vector_scores[idx] = 1 - (dist / 2)

        # ── Combine ──
        combined = (self.alpha * vector_scores) + ((1 - self.alpha) * bm25_scores)
        top_indices = np.argsort(combined)[::-1][:n_results]
        return [self.chunks[i] for i in top_indices]