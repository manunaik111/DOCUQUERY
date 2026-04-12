from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from rag.query_optimizer import rewrite_query
import os
from dotenv import load_dotenv

load_dotenv()

# ── State ─────────────────────────────────────────────────────
class RAGState(TypedDict):
    query: str
    optimized_query: str
    chunks: list
    answer: str
    final_answer: str
    retrieval_relevant: bool
    answer_grounded: bool
    retry_count: int
    searcher: object

def get_llm():
    return ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.1,
        groq_api_key=os.getenv('GROQ_API_KEY')
    )

# ── Node 1: Query Optimizer ───────────────────────────────────
def optimize_query_node(state: RAGState) -> RAGState:
    optimized = rewrite_query(state['query'])
    return {**state, 'optimized_query': optimized}

# ── Node 2: Retrieve ─────────────────────────────────────────
def retrieve_node(state: RAGState) -> RAGState:
    from rag.reranker import rerank
    searcher = state['searcher']
    candidates = searcher.search(state['optimized_query'], n_results=20)
    top_chunks = rerank(state['optimized_query'], candidates, top_k=3)
    return {**state, 'chunks': top_chunks}

# ── Node 3: Evaluate Retrieval ────────────────────────────────
def evaluate_retrieval_node(state: RAGState) -> RAGState:
    llm = get_llm()
    context = '\n'.join([c['text'][:300] for c in state['chunks']])
    response = llm.invoke([
        SystemMessage(content='''You are a retrieval evaluator.
Answer ONLY with YES or NO.
YES = the retrieved chunks contain information relevant to answering the query.
NO = the chunks are off-topic or don't address the query.'''),
        HumanMessage(content=f'Query: {state["optimized_query"]}\n\nChunks:\n{context}')
    ])
    relevant = 'YES' in response.content.upper()
    return {**state, 'retrieval_relevant': relevant}

# ── Node 4: Generate Answer ───────────────────────────────────
def generate_answer_node(state: RAGState) -> RAGState:
    llm = get_llm()
    context = '\n\n'.join([
        f'[Page {c["page"]}]: {c["text"]}' for c in state['chunks']
    ])
    response = llm.invoke([
        SystemMessage(content='''You are DocuQuery.
Answer ONLY from the provided context.
If not found, say: I could not find this in the document.
Cite page numbers.'''),
        HumanMessage(content=f'Context:\n{context}\n\nQuestion: {state["query"]}')
    ])
    return {**state, 'answer': response.content}

# ── Node 5: Hallucination Check ───────────────────────────────
def check_hallucination_node(state: RAGState) -> RAGState:
    llm = get_llm()
    context = '\n'.join([c['text'] for c in state['chunks']])
    response = llm.invoke([
        SystemMessage(content='''You are a fact-checker.
Answer ONLY with YES or NO.
YES = every claim in the answer is directly supported by the context.
NO = the answer contains claims not found in the context.'''),
        HumanMessage(content=f'Context:\n{context}\n\nAnswer:\n{state["answer"]}')
    ])
    grounded = 'YES' in response.content.upper()
    return {**state, 'answer_grounded': grounded}

# ── Node 6: Final Output ──────────────────────────────────────
def final_output_node(state: RAGState) -> RAGState:
    return {**state, 'final_answer': state['answer']}

# ── Conditional Edges ─────────────────────────────────────────
def route_retrieval(state: RAGState) -> str:
    if state['retrieval_relevant'] or state.get('retry_count', 0) >= 1:
        return 'generate'
    return 'retry'

def route_hallucination(state: RAGState) -> str:
    if state['answer_grounded'] or state.get('retry_count', 0) >= 1:
        return 'output'
    return 'regenerate'

# ── Build Graph ───────────────────────────────────────────────
def build_self_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node('optimize', optimize_query_node)
    graph.add_node('retrieve', retrieve_node)
    graph.add_node('evaluate_retrieval', evaluate_retrieval_node)
    graph.add_node('generate', generate_answer_node)
    graph.add_node('check_hallucination', check_hallucination_node)
    graph.add_node('output', final_output_node)

    graph.set_entry_point('optimize')
    graph.add_edge('optimize', 'retrieve')
    graph.add_edge('retrieve', 'evaluate_retrieval')
    graph.add_conditional_edges('evaluate_retrieval', route_retrieval, {
        'generate': 'generate',
        'retry': 'optimize'
    })
    graph.add_edge('generate', 'check_hallucination')
    graph.add_conditional_edges('check_hallucination', route_hallucination, {
        'output': 'output',
        'regenerate': 'generate'
    })
    graph.add_edge('output', END)

    return graph.compile()

# ── Run ───────────────────────────────────────────────────────
def run_self_rag(query: str, searcher) -> dict:
    graph = build_self_rag_graph()
    initial_state = RAGState(
        query=query,
        optimized_query='',
        chunks=[],
        answer='',
        final_answer='',
        retrieval_relevant=False,
        answer_grounded=False,
        retry_count=0,
        searcher=searcher
    )
    result = graph.invoke(initial_state)
    return {
        'answer': result['final_answer'],
        'chunks': result['chunks'],
        'optimized_query': result['optimized_query'],
        'source_pages': sorted(set(c['page'] for c in result['chunks']))
    }