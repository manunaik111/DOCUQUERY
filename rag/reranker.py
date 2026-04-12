from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker


def rerank(query: str, candidates: list[dict], top_k=3) -> list[dict]:
    '''
    candidates: list of {'text': str, 'page': int}
    Returns: top_k re-ranked candidates
    '''
    reranker = get_reranker()
    pairs = [(query, c['text']) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        {**chunk, 'rerank_score': float(score)}
        for score, chunk in ranked[:top_k]
    ]