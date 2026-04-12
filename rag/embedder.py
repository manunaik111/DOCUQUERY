import chromadb
from chromadb.utils import embedding_functions

def build_vectorstore(chunks: list[dict]):
    '''
    chunks: list of {'text': str, 'page': int}
    Returns: ChromaDB collection
    '''
    client = chromadb.Client()
    
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Fresh collection each time a new PDF is uploaded
    try:
        client.delete_collection("docuquery")
    except:
        pass
    
    collection = client.create_collection(
        name="docuquery",
        embedding_function=ef
    )
    
    collection.add(
        documents=[c['text'] for c in chunks],
        metadatas=[{'page': c['page']} for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    return collection


def retrieve(collection, query: str, n_results=5) -> list[dict]:
    '''
    Returns top n_results chunks for the query.
    '''
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    chunks = []
    for text, meta in zip(results['documents'][0], results['metadatas'][0]):
        chunks.append({
            'text': text,
            'page': meta['page']
        })
    
    return chunks