import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def get_llm():
    return ChatGroq(
        model='llama-3.1-8b-instant',
        temperature=0.1,
        groq_api_key=os.getenv('GROQ_API_KEY')
    )

def generate_answer(query: str, chunks: list[dict]) -> dict:
    '''
    chunks: list of {'text': str, 'page': int}
    Returns: {'answer': str, 'source_pages': list[int]}
    '''
    llm = get_llm()

    context = "\n\n".join([
        f"[Page {c['page']}]: {c['text']}"
        for c in chunks
    ])

    response = llm.invoke([
        SystemMessage(content='''You are DocuQuery, a document assistant.
Answer ONLY from the provided context.
If the answer is not in the context, say: I could not find this in the document.
Always cite the page numbers you used.'''),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ])

    source_pages = sorted(set(c['page'] for c in chunks))

    return {
        'answer': response.content,
        'source_pages': source_pages
    }