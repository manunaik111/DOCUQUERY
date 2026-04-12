import streamlit as st
from rag.loader import load_pdf
from rag.embedder import build_vectorstore
from rag.parent_retriever import build_parent_retriever
from rag.hybrid_search import HybridSearcher
from rag.self_rag import run_self_rag

st.set_page_config(page_title="DocuQuery", page_icon=None, layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&family=DM+Sans:wght@400;500;700&display=swap');

html, body, .stApp {
    background-color: #F5F0E8 !important;
    font-family: 'DM Sans', sans-serif;
    color: #0A0A0A;
}

section[data-testid="stSidebar"] {
    background-color: #F5F0E8 !important;
    border-right: 4px solid #0A0A0A !important;
}

section[data-testid="stSidebar"] * {
    color: #0A0A0A !important;
}

section[data-testid="stSidebar"] .stMarkdown h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.8rem !important;
    letter-spacing: 0.08em;
}

[data-testid="stFileUploaderDropzone"] {
    border-radius: 0 !important;
    border: 3px dashed #0A0A0A !important;
    background: #fff !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    background: #ede8e0 !important;
}

header[data-testid="stHeader"] {
    background-color: #F5F0E8 !important;
    border-bottom: 3px solid #0A0A0A !important;
}

[data-testid="stChatMessage"] {
    border: 3px solid #0A0A0A !important;
    border-radius: 0 !important;
    box-shadow: 5px 5px 0px #0A0A0A;
    margin-bottom: 1rem;
    background: #fff !important;
}

[data-testid="stChatInput"] {
    border: 3px solid #0A0A0A !important;
    border-radius: 0 !important;
    background: #fff !important;
    box-shadow: 5px 5px 0px #0A0A0A;
}

[data-testid="stChatInput"] textarea {
    border-radius: 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    background: #fff !important;
}

[data-testid="stExpander"] {
    border: 3px solid #0A0A0A !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0px #0A0A0A;
    background: #F5F0E8 !important;
}

[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

[data-testid="stAlert"] {
    border-radius: 0 !important;
    border: 2px solid #0A0A0A !important;
    background: #fff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
}

hr {
    border: 2px solid #0A0A0A !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F5F0E8; }
::-webkit-scrollbar-thumb { background: #0A0A0A; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if 'searcher' not in st.session_state:
    st.session_state.searcher = None
if 'filename' not in st.session_state:
    st.session_state.filename = None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# DOCUQUERY")
    st.markdown("---")
    st.markdown("**UPLOAD DOCUMENT**")

    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file and uploaded_file.name != st.session_state.filename:
        with st.spinner("Indexing..."):
            chunks = load_pdf(uploaded_file)
            collection = build_vectorstore(chunks)
            page_texts = [c['text'] for c in chunks]
            build_parent_retriever(page_texts)
            st.session_state.searcher = HybridSearcher(chunks, collection, alpha=0.5)
            st.session_state.filename = uploaded_file.name
        st.success(f"Indexed {len(chunks)} pages")

# ── Main ──────────────────────────────────────────────────────
st.markdown("""
<div style="font-family: 'Bebas Neue', sans-serif; font-size: 4rem; letter-spacing: 0.05em; line-height: 1; color: #0A0A0A;">
    DOCUQUERY
</div>
<div style="font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: #666; margin-top: 0.3rem;">
    Document Intelligence System &mdash; v2.0
</div>
<hr style="border: 2px solid #0A0A0A; margin: 1.5rem 0;">
""", unsafe_allow_html=True)

if st.session_state.searcher is None:
    st.markdown("""
    <div style="border: 4px solid #0A0A0A; padding: 2.5rem; background: #fff; box-shadow: 8px 8px 0px #0A0A0A; max-width: 600px;">
        <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            NO DOCUMENT LOADED
        </div>
        <div style="font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #444; line-height: 1.8;">
            Upload a PDF from the sidebar to begin.<br>
            Supports any text-based PDF document.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(
        f'<div style="display: inline-block; background: #0A0A0A; color: #F5F0E8; font-family: Space Mono, monospace; font-size: 0.72rem; padding: 0.4rem 1rem; margin-bottom: 1.5rem; letter-spacing: 0.05em;">ACTIVE &mdash; {st.session_state.filename}</div>',
        unsafe_allow_html=True
    )

    query = st.chat_input("Ask a question about your document...")

    if query:
        with st.spinner("Running Self-RAG pipeline..."):
            result = run_self_rag(query, st.session_state.searcher)

        st.chat_message("user").write(query)
        st.chat_message("assistant").write(result['answer'])

        with st.expander("QUERY OPTIMIZATION"):
            st.markdown(
                f'<p style="font-family: Space Mono, monospace; font-size: 0.72rem;"><strong>ORIGINAL</strong><br>{query}</p>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<p style="font-family: Space Mono, monospace; font-size: 0.72rem; margin-top: 0.8rem;"><strong>OPTIMIZED</strong><br>{result["optimized_query"]}</p>',
                unsafe_allow_html=True
            )

        with st.expander("SOURCE PAGES"):
            st.markdown(
                f'<span style="font-family: Space Mono, monospace; font-size: 0.75rem;">Pages referenced: {result["source_pages"]}</span>',
                unsafe_allow_html=True
            )
            for c in result['chunks']:
                score = c.get('rerank_score', 0)
                st.markdown(
                    f'<p style="font-family: Space Mono, monospace; font-size: 0.72rem; margin-top: 0.8rem;"><strong>Page {c["page"]} &mdash; score: {score:.3f}</strong><br>{c["text"][:300]}...</p>',
                    unsafe_allow_html=True
                )