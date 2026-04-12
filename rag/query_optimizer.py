import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def get_llm():
    return ChatGroq(
        model='llama-3.1-8b-instant',
        temperature=0,
        groq_api_key=os.getenv('GROQ_API_KEY')
    )

def rewrite_query(original_query: str) -> str:
    '''
    Rewrites a messy user query into a clean retrieval-optimized query.
    '''
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content='''You are a query optimization assistant.
Rewrite the user's question into a clear, specific, retrieval-optimized query.
Return ONLY the rewritten query. No explanations. No preamble.
Keep it under 30 words.'''),
        HumanMessage(content=f'Original query: {original_query}')
    ])

    return response.content.strip()


def hyde_query(original_query: str) -> str:
    '''
    HyDE: generates a hypothetical document passage to use for retrieval.
    '''
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content='''You are a document assistant.
Write a short hypothetical document passage (2-3 sentences) that would
perfectly answer the user's question.
Write it as if it's a direct excerpt from a real document.
Return ONLY the passage. No preamble.'''),
        HumanMessage(content=f'Question: {original_query}')
    ])

    return response.content.strip()