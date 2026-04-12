from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class ParentRetriever:
    def __init__(self, chunks_text: list[str]):
        self.parent_chunks = []
        self.child_to_parent = {}

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150, chunk_overlap=20
        )

        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        all_children = []
        all_child_ids = []

        for text in chunks_text:
            parents = parent_splitter.split_text(text)
            for parent_text in parents:
                parent_idx = len(self.parent_chunks)
                self.parent_chunks.append(parent_text)

                children = child_splitter.split_text(parent_text)
                for child_text in children:
                    child_id = f"child_{len(all_children)}"
                    self.child_to_parent[child_id] = parent_idx
                    all_children.append(child_text)
                    all_child_ids.append(child_id)

        docs = [
            Document(page_content=c, metadata={"child_id": cid})
            for c, cid in zip(all_children, all_child_ids)
        ]

        self.vectorstore = Chroma.from_documents(
            docs,
            embedding=embeddings,
            collection_name="child_chunks"
        )

    def invoke(self, query: str, k=4) -> list[str]:
        results = self.vectorstore.similarity_search(query, k=k)
        seen = set()
        parents = []
        for doc in results:
            child_id = doc.metadata.get('child_id', '')
            parent_idx = self.child_to_parent.get(child_id)
            if parent_idx is not None and parent_idx not in seen:
                seen.add(parent_idx)
                parents.append(self.parent_chunks[parent_idx])
        return parents


def build_parent_retriever(chunks_text: list[str]) -> ParentRetriever:
    return ParentRetriever(chunks_text)


def retrieve_with_parent(retriever: ParentRetriever, query: str) -> list[dict]:
    results = retriever.invoke(query)
    return [{'text': text, 'page': 0} for text in results]