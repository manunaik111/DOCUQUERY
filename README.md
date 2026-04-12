<div align="center">

```
██████╗  ██████╗  ██████╗██╗   ██╗ ██████╗ ██╗   ██╗███████╗██████╗ ██╗   ██╗
██╔══██╗██╔═══██╗██╔════╝██║   ██║██╔═══██╗██║   ██║██╔════╝██╔══██╗╚██╗ ██╔╝
██║  ██║██║   ██║██║     ██║   ██║██║   ██║██║   ██║█████╗  ██████╔╝ ╚████╔╝ 
██║  ██║██║   ██║██║     ██║   ██║██║▄▄ ██║██║   ██║██╔══╝  ██╔══██╗  ╚██╔╝  
██████╔╝╚██████╔╝╚██████╗╚██████╔╝╚██████╔╝╚██████╔╝███████╗██║  ██║   ██║   
╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝  
```

**Document Intelligence System — v2.0**

[![Python](https://img.shields.io/badge/Python-3.10%2B-black?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-black?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1%2B-black?style=flat-square&logo=langchain&logoColor=white)](https://langgraph.dev)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-black?style=flat-square)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-black?style=flat-square)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-black?style=flat-square)](LICENSE)

*Upload any PDF. Ask anything. Get grounded, hallucination-checked answers — with source pages.*

</div>

---

## What Is DocuQuery?

**DocuQuery** is a production-grade, **Self-RAG** (Retrieval-Augmented Generation) document intelligence system. It goes far beyond a simple "chat with your PDF" app — it implements a full **agentic pipeline** using LangGraph that reasons about retrieval quality, detects hallucinations, and self-corrects before returning an answer.

Built with a brutalist, editorial design aesthetic and backed by a 5-stage AI pipeline, DocuQuery is engineered for serious document analysis — not toy demos.

---

## ✦ Key Features

| Feature | Description |
|---|---|
| 🔁 **Self-RAG Pipeline** | A LangGraph agentic graph that loops, retries, and self-corrects at every stage |
| 🔍 **Hybrid Search** | Combines dense vector search (ChromaDB) + sparse BM25 with a configurable alpha blend |
| 📐 **Parent-Child Retrieval** | Embeds small child chunks for precision, returns large parent chunks for full context |
| ✍️ **LLM Query Optimization** | Rewrites vague user queries into clean, retrieval-optimized queries before searching |
| 🎯 **Cross-Encoder Reranking** | MS-MARCO cross-encoder re-scores top candidates for maximum relevance |
| 🛡️ **Hallucination Guard** | An LLM fact-checker verifies every answer is grounded in retrieved context |
| 📄 **Source Page Citations** | Every answer cites the exact PDF page numbers it used |
| 🔬 **Full Pipeline Transparency** | Expanders show the optimized query, source pages, and per-chunk rerank scores |

---

## ⚙️ Architecture

DocuQuery runs a **6-node agentic graph** built with LangGraph. Every user question flows through this graph before an answer is returned.

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH STATE MACHINE                      │
│                                                                     │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │   Node 1    │────▶│    Node 2    │────▶│       Node 3         │ │
│  │  OPTIMIZE   │     │   RETRIEVE   │     │ EVALUATE RETRIEVAL   │ │
│  │             │     │              │     │                      │ │
│  │ LLM rewrites│     │ Hybrid Search│     │ LLM grades: are the  │ │
│  │ query for   │     │ (BM25 +      │     │ chunks relevant to   │ │
│  │ retrieval   │     │  ChromaDB)   │     │ the query? YES/NO    │ │
│  │             │     │ → CrossEncoder│    │                      │ │
│  │             │     │   Reranker   │     │                      │ │
│  └─────────────┘     └──────────────┘     └──────────┬───────────┘ │
│        ▲                                              │             │
│        │                    NO (retry)  ◀─────────────┘             │
│        └───────────────────────────────              │ YES          │
│                                                      ▼             │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │   Node 6    │◀────│    Node 5    │◀────│       Node 4         │ │
│  │   OUTPUT    │     │   HALLUCIN-  │     │      GENERATE        │ │
│  │             │     │   ATION      │     │                      │ │
│  │ Return final│     │   CHECK      │     │ LLaMA 3.3 70B        │ │
│  │ answer +    │     │              │     │ generates answer     │ │
│  │ metadata    │     │ LLM verifies │     │ from top-3 chunks    │ │
│  │             │     │ answer is    │     │ with page citations  │ │
│  │             │     │ grounded:    │     │                      │ │
│  │             │     │ YES/NO       │     │                      │ │
│  └─────────────┘     └──────┬───────┘     └──────────────────────┘ │
│                             │                                       │
│              NO (regenerate)└──────────────────▶ Node 4            │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
Answer + Optimized Query + Source Pages + Rerank Scores
```

### Retrieval Stack (Depth-First)

```
PDF Upload
    │
    ├── PyMuPDF (fitz)
    │     └── Page-by-page text extraction → [{text, page}]
    │
    ├── ChromaDB (in-memory)
    │     └── all-MiniLM-L6-v2 embeddings → dense vector store
    │
    ├── BM25Okapi (rank-bm25)
    │     └── tokenized corpus → sparse keyword index
    │
    └── ParentRetriever (LangChain)
          ├── Parent chunks: 1500 chars, 100 overlap
          └── Child chunks:  150 chars, 20 overlap
                └── all-MiniLM-L6-v2 → child Chroma store
```

---

## 🗂️ Project Structure

```
docuquery/
│
├── app.py                    # Streamlit UI + session state management
│
├── rag/
│   ├── __init__.py
│   ├── loader.py             # PDF → page chunks (PyMuPDF)
│   ├── embedder.py           # ChromaDB vector store builder
│   ├── hybrid_search.py      # BM25 + vector hybrid searcher (alpha blend)
│   ├── parent_retriever.py   # Parent-child chunk retriever (LangChain)
│   ├── reranker.py           # Cross-encoder reranker (MS-MARCO MiniLM)
│   ├── query_optimizer.py    # LLM query rewriter + HyDE generator
│   ├── self_rag.py           # LangGraph Self-RAG graph (6-node)
│   └── chain.py              # Base LLM chain (ChatGroq + context builder)
│
├── .env                      # API keys (never committed)
├── .env.example              # Template for environment setup
├── .gitignore                # Comprehensive ignore rules
├── requirements.txt          # All pinned dependencies
└── README.md                 # You are here
```

---

## 🧩 Module Reference

### `rag/loader.py` — PDF Loader
Accepts a Streamlit `UploadedFile` and returns a list of `{text, page}` dicts using **PyMuPDF** (`fitz`). Blank pages are skipped automatically.

```python
pages = load_pdf(uploaded_file)
# → [{'text': '...', 'page': 1}, {'text': '...', 'page': 2}, ...]
```

---

### `rag/embedder.py` — Vector Store Builder
Builds an **in-memory ChromaDB collection** with `all-MiniLM-L6-v2` embeddings. A fresh collection is created on every new PDF upload to prevent cross-document contamination.

```python
collection = build_vectorstore(chunks)
# → ChromaDB collection ready for .query()
```

---

### `rag/hybrid_search.py` — Hybrid Searcher
Combines **dense vector scores** (ChromaDB cosine similarity) and **sparse BM25 scores** into a single weighted ranking:

```
combined_score = α × vector_score + (1 - α) × bm25_score
```

Default `alpha=0.5` gives equal weight. Set `alpha=1.0` for pure vector, `0.0` for pure BM25.

```python
searcher = HybridSearcher(chunks, collection, alpha=0.5)
results = searcher.search("what is the refund policy?", n_results=20)
```

---

### `rag/parent_retriever.py` — Parent-Child Retriever
Implements the **Parent Document Retriever** pattern:
- **Child chunks** (150 chars) are embedded and indexed for high-precision similarity matching
- When a child chunk matches, its **parent chunk** (1500 chars) is returned — giving the LLM full context

This solves the classic RAG tradeoff: small chunks for retrieval accuracy, large chunks for answer quality.

---

### `rag/reranker.py` — Cross-Encoder Reranker
Re-scores the top 20 hybrid search candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike bi-encoders, cross-encoders jointly encode the query+passage pair for maximum relevance accuracy.

```python
top_chunks = rerank(query, candidates, top_k=3)
# → [{...chunk..., 'rerank_score': 0.943}, ...]
```

The model is **lazily loaded** (singleton pattern) — only initialized on first use, then cached.

---

### `rag/query_optimizer.py` — Query Rewriter + HyDE
Two LLM-powered query enhancement strategies:

**`rewrite_query()`** — Rewrites a vague user question into a clean, retrieval-optimized query using LLaMA 3.3 70B at temperature 0.

**`hyde_query()`** — Implements **HyDE** (Hypothetical Document Embeddings): generates a fake-but-plausible document passage that would answer the question, then uses it as the retrieval query. This dramatically improves retrieval for abstract or complex questions.

---

### `rag/self_rag.py` — LangGraph Self-RAG Graph
The core of DocuQuery. A **stateful, conditional graph** with 6 nodes and retry loops:

| Node | Role |
|---|---|
| `optimize` | Rewrites query via LLM |
| `retrieve` | Runs hybrid search → cross-encoder rerank |
| `evaluate_retrieval` | LLM grades retrieval relevance (YES/NO) |
| `generate` | LLaMA 3.3 70B generates answer from top-3 chunks |
| `check_hallucination` | LLM verifies every claim is grounded in context |
| `output` | Returns final answer + metadata |

**Retry logic:**
- If retrieval is irrelevant → re-optimize query and re-retrieve (max 1 retry)
- If answer is hallucinated → regenerate from same chunks (max 1 retry)

---

### `rag/chain.py` — Base LLM Chain
Standalone answer generator using `ChatGroq` + `LangChain` message formatting. Used as the base LLM utility across the pipeline.

---

### `app.py` — Streamlit UI
The main application with:
- **Brutalist editorial design** — cream background (`#F5F0E8`), hard black borders, box-shadows
- **Bebas Neue** headers, **Space Mono** monospace labels, **DM Sans** body text
- Sidebar PDF uploader with live indexing progress
- Chat-style Q&A interface
- Expandable **Query Optimization** panel (original vs optimized query)
- Expandable **Source Pages** panel with per-chunk scores and text previews

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit 1.32+ |
| **LLM** | LLaMA 3.3 70B Versatile via Groq API |
| **Orchestration** | LangGraph 0.1+ (StateGraph) |
| **LLM Framework** | LangChain Core, LangChain Community, LangChain Groq |
| **Vector Store** | ChromaDB (in-memory) |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) |
| **Sparse Retrieval** | BM25Okapi (rank-bm25) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Parent Retrieval** | LangChain `RecursiveCharacterTextSplitter` + Chroma |
| **Environment** | python-dotenv |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/docuquery.git
cd docuquery
```

### 2. Create & Activate Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will also download the `all-MiniLM-L6-v2` embedding model (~90MB) and the `ms-marco-MiniLM-L-6-v2` cross-encoder (~85MB) from Hugging Face. These are cached locally after the first download.

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com). Groq provides extremely fast inference for LLaMA models at no cost.

### 5. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 Usage Guide

1. **Upload a PDF** using the sidebar uploader — any text-based PDF works
2. Wait for indexing — the app extracts text, builds the vector store, and indexes BM25 simultaneously
3. **Ask a question** in the chat input at the bottom
4. View the **answer** with page citations in the chat area
5. Expand **QUERY OPTIMIZATION** to see how your query was rewritten
6. Expand **SOURCE PAGES** to inspect the exact chunks used, with their rerank scores

---

## 🔬 How Self-RAG Works (Under the Hood)

Traditional RAG retrieves once and generates once. **Self-RAG adds two critical feedback loops:**

**Loop 1 — Retrieval Evaluation:**
After retrieval, an LLM evaluates whether the chunks actually contain information relevant to the query. If chunks are off-topic (e.g., BM25 matched noise words), the query is re-optimized and retrieval runs again.

**Loop 2 — Hallucination Check:**
After generation, a second LLM call verifies that every factual claim in the answer is directly supported by the retrieved chunks. If the answer introduces details not found in the context, the model regenerates.

Both loops have a **retry cap of 1** to prevent infinite loops in edge cases.

---

## 🛡️ Environment & Security

- `.env` is **git-ignored** — your API key is never committed
- ChromaDB runs **entirely in-memory** — no data persists after the session ends
- No uploaded documents are stored on disk — all processing happens in-memory via Streamlit's `UploadedFile` buffer

---

## 📦 dependencies

```
streamlit>=1.32.0          # Web UI framework
langchain>=0.2.0           # LLM orchestration
langchain-core>=0.2.0      # Core message types & document schema
langchain-community>=0.2.0 # HuggingFaceEmbeddings, Chroma vectorstore
langchain-groq>=0.1.0      # Groq ChatGroq LLM integration
langgraph>=0.1.0           # Agentic state graph
chromadb>=0.4.22           # In-memory vector database
sentence-transformers>=2.5.0  # MiniLM embeddings + cross-encoder reranker
pymupdf>=1.23.0            # PDF text extraction (fitz)
rank-bm25>=0.2.2           # BM25Okapi sparse search
python-dotenv>=1.0.0       # .env file loader
```

---

## 🗺️ Roadmap

- [ ] Multi-document support (query across multiple PDFs simultaneously)
- [ ] HyDE integration into the main retrieval path
- [ ] Persistent ChromaDB storage option for large documents
- [ ] Streaming LLM responses
- [ ] Export answers + citations as PDF report
- [ ] Structured output extraction (tables, entities)

---

## 👤 Author

Built by **Manu Naik** — AI & ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-black?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME)

---

<div align="center">

*"The best RAG system is one that knows when it's wrong."*

**DocuQuery — Document Intelligence System v2.0**

</div>
