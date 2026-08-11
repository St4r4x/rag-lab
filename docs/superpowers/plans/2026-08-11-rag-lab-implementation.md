# rag-lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a learning RAG assistant that answers questions about the LangChain and LangGraph documentation, starting as a linear LangChain/LangGraph pipeline and growing — on the same graph — into an agentic RAG with self-correction.

**Architecture:** A single LangGraph `StateGraph` that starts as `retrieve → generate` (v1) and gains `grade_documents` / `rewrite_query` nodes plus a bounded retry loop (v2) without ever being rewritten. Ingestion is a standalone offline script that populates a Qdrant collection. A FastAPI service wraps the graph; a Gradio UI talks to that service over HTTP — it never calls the graph directly.

**Tech Stack:** Python 3.11+, LangChain (`init_chat_model`/`init_embeddings`), LangGraph, Qdrant (`langchain-qdrant`, Docker Compose), FastAPI, Gradio, pytest.

## Global Constraints

- LLM and embeddings are obtained only via `init_chat_model()` / `init_embeddings()`, configured through env vars `LLM_MODEL` / `EMBEDDING_MODEL` — no custom provider-abstraction layer.
- Vector store is Qdrant via Docker Compose, single collection `langchain_docs`.
- Exactly one `StateGraph` evolves from v1 (`retrieve → generate`) to v2 (adds `grade_documents` / `rewrite_query` / a loop bounded to 2 retries) — never two separate pipelines.
- The Gradio UI calls the FastAPI service over HTTP; it never imports or invokes the graph directly.
- Test coverage is scoped to the v2 graph routing logic only (mocked LLM/vector store) — no unit tests for ingestion, the v1 graph, the API, or the UI. This is an approved YAGNI decision from the spec, not an oversight.
- No silent retries on provider or Qdrant errors — propagate clearly (FastAPI `502`); `/health` checks Qdrant at startup.
- Python dependencies live in a local `.venv` at the project root (`python3 -m venv .venv`), never in the system environment; `.venv/` is gitignored.
- Git commits: English, imperative mood, Conventional Commits prefix (`feat|fix|docs|chore|refactor|test|ci|style`), subject ≤72 chars, no trailing period.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `docker-compose.yml`

**Interfaces:**
- Produces: an installable project (`pip install -e ".[dev]"`) and a running Qdrant container on `localhost:6333`. No Python symbols yet — later tasks import from packages declared here.

- [ ] **Step 1: Create the virtualenv**

Run: `cd ~/Projects/rag-lab && python3 -m venv .venv`

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "rag-lab"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3.9",
    "langgraph>=0.2.60",
    "langchain-community>=0.3.0",
    "langchain-text-splitters>=0.3.0",
    "langchain-ollama>=0.2.0",
    "langchain-openai>=0.2.0",
    "langchain-qdrant>=0.2.0",
    "qdrant-client>=1.12.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "gradio>=5.0.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["ingestion*", "graph*", "api*", "ui*"]
```

- [ ] **Step 3: Install dependencies into the venv**

Run: `.venv/bin/pip install -e ".[dev]"`
Expected: install completes with no errors.

- [ ] **Step 4: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.env
```

- [ ] **Step 5: Write `.env.example`**

```
LLM_MODEL=ollama:llama3.1
EMBEDDING_MODEL=ollama:nomic-embed-text
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=langchain_docs
RAG_API_URL=http://localhost:8000

# To use an API provider instead of Ollama, uncomment and set a key:
# LLM_MODEL=openai:gpt-4o-mini
# EMBEDDING_MODEL=openai:text-embedding-3-small
# OPENAI_API_KEY=sk-...
```

- [ ] **Step 6: Write `docker-compose.yml`**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

- [ ] **Step 7: Verify Qdrant starts and is reachable**

Run: `docker compose up -d && sleep 3 && curl -s http://localhost:6333/collections`
Expected: `{"result":{"collections":[]},"status":"ok","time":...}`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore .env.example docker-compose.yml
git commit -m "chore: scaffold rag-lab project"
```

---

## Task 2: Config module (LLM, embeddings, vector store factories)

**Files:**
- Create: `config.py`

**Interfaces:**
- Consumes: env vars `LLM_MODEL`, `EMBEDDING_MODEL`, `QDRANT_URL`, `QDRANT_COLLECTION` (all optional, defaults below).
- Produces: `get_llm() -> BaseChatModel`, `get_embeddings() -> Embeddings`, `get_vectorstore() -> QdrantVectorStore`, and module constants `QDRANT_URL: str`, `QDRANT_COLLECTION: str`. Tasks 3, 4, 5, and 6 all import from this module.

- [ ] **Step 1: Write `config.py`**

```python
import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "langchain_docs")


def get_llm():
    model = os.environ.get("LLM_MODEL", "ollama:llama3.1")
    return init_chat_model(model)


def get_embeddings():
    model = os.environ.get("EMBEDDING_MODEL", "ollama:nomic-embed-text")
    return init_embeddings(model)


@lru_cache
def get_vectorstore() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=get_embeddings(),
    )
```

- [ ] **Step 2: Verify the LLM and embeddings factories work**

Prerequisite: `ollama serve` running locally with `ollama pull llama3.1` and `ollama pull nomic-embed-text` done.

Run:
```bash
.venv/bin/python -c "
from config import get_llm, get_embeddings
print(get_llm().invoke('Say OK.').content)
print(len(get_embeddings().embed_query('hello')))
"
```
Expected: prints a short LLM reply, then an integer (the embedding dimension, e.g. `768`).

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add LLM, embeddings and vector store config"
```

---

## Task 3: Ingestion script

**Files:**
- Create: `ingestion/__init__.py` (empty)
- Create: `ingestion/ingest.py`

**Interfaces:**
- Consumes: `config.get_embeddings()`, `config.QDRANT_URL`, `config.QDRANT_COLLECTION`.
- Produces: a populated Qdrant collection (`langchain_docs`) with document chunks and metadata `{"source": "langchain"|"langgraph", "path": str}`. Runnable as `python -m ingestion.ingest`. Task 4's manual verification depends on this collection existing.

- [ ] **Step 1: Write `ingestion/__init__.py`**

Empty file.

- [ ] **Step 2: Write `ingestion/ingest.py`**

```python
import subprocess
import tempfile
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import QDRANT_COLLECTION, QDRANT_URL, get_embeddings

REPOS = {
    "langchain": "https://github.com/langchain-ai/langchain.git",
    "langgraph": "https://github.com/langchain-ai/langgraph.git",
}


def clone_docs(repo_url: str, dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
        capture_output=True,
    )


def load_markdown_files(repo_dir: Path, source: str) -> list[dict]:
    docs_dir = repo_dir / "docs"
    files = list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.mdx"))
    pages = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            pages.append({"text": text, "source": source, "path": str(f.relative_to(repo_dir))})
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for page in pages:
        for chunk_text in splitter.split_text(page["text"]):
            chunks.append({"text": chunk_text, "source": page["source"], "path": page["path"]})
    return chunks


def main() -> None:
    embeddings = get_embeddings()
    all_chunks: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        for source, repo_url in REPOS.items():
            dest = Path(tmp) / source
            clone_docs(repo_url, dest)
            all_chunks.extend(chunk_pages(load_markdown_files(dest, source)))

    texts = [c["text"] for c in all_chunks]
    metadatas = [{"source": c["source"], "path": c["path"]} for c in all_chunks]

    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
    )
    print(f"Ingested {len(texts)} chunks into '{QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run ingestion and verify**

Run: `.venv/bin/python -m ingestion.ingest`
Expected: prints `Ingested <N> chunks into 'langchain_docs'.` with `N` in the thousands (clones both doc trees).

Run: `curl -s http://localhost:6333/collections/langchain_docs | python3 -m json.tool | grep points_count`
Expected: `points_count` matches `N` from the previous step.

- [ ] **Step 4: Commit**

```bash
git add ingestion/
git commit -m "feat: add LangChain/LangGraph docs ingestion script"
```

---

## Task 4: Graph v1 — retrieve → generate

**Files:**
- Create: `graph/__init__.py` (empty)
- Create: `graph/state.py`
- Create: `graph/build.py`

**Interfaces:**
- Consumes: `config.get_llm()`, `config.get_vectorstore()`.
- Produces: `RAGState` (TypedDict: `question: str`, `documents: list[Document]`, `generation: str`, `retries: int`) in `graph/state.py`; `make_retrieve(vectorstore)`, `make_generate(llm)`, `build_graph_v1(llm, vectorstore)` in `graph/build.py`. Task 5 extends `graph/build.py` and reuses `RAGState`, `make_retrieve`, `make_generate`.

- [ ] **Step 1: Write `graph/__init__.py`**

Empty file.

- [ ] **Step 2: Write `graph/state.py`**

```python
from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    retries: int
```

- [ ] **Step 3: Write `graph/build.py` (v1 only)**

```python
from langgraph.graph import END, START, StateGraph

from graph.state import RAGState


def make_retrieve(vectorstore):
    def retrieve(state: RAGState) -> dict:
        docs = vectorstore.similarity_search(state["question"], k=4)
        return {"documents": docs}

    return retrieve


def make_generate(llm):
    def generate(state: RAGState) -> dict:
        if not state["documents"]:
            return {"generation": "Je n'ai pas assez d'information dans la documentation indexée pour répondre."}
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        response = llm.invoke(
            "Answer the question using only the context below. "
            "If the context is insufficient, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['question']}"
        )
        return {"generation": response.content}

    return generate


def build_graph_v1(llm, vectorstore):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve(vectorstore))
    graph.add_node("generate", make_generate(llm))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
```

- [ ] **Step 4: Verify manually against the ingested docs**

Run:
```bash
.venv/bin/python -c "
from config import get_llm, get_vectorstore
from graph.build import build_graph_v1

graph = build_graph_v1(get_llm(), get_vectorstore())
result = graph.invoke({'question': 'What is a StateGraph in LangGraph?', 'documents': [], 'generation': '', 'retries': 0})
print(result['generation'])
"
```
Expected: a plausible answer mentioning LangGraph's `StateGraph` (grounded in the ingested docs, not empty).

- [ ] **Step 5: Commit**

```bash
git add graph/
git commit -m "feat: add v1 retrieve-generate graph"
```

---

## Task 5: Graph v2 — grade, rewrite, bounded retry loop + routing test

**Files:**
- Modify: `graph/build.py` (add v2 nodes and `build_graph_v2`, alongside the existing v1 code from Task 4)
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_graph_routing.py`

**Interfaces:**
- Consumes: `RAGState`, `make_retrieve`, `make_generate` from Task 4.
- Produces: `MAX_RETRIES = 2`, `make_grade_documents(llm)`, `make_rewrite_query(llm)`, `route_after_grade(state) -> str`, `build_graph_v2(llm, vectorstore)` — all in `graph/build.py`. Task 6 imports `build_graph_v2`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_routing.py
from types import SimpleNamespace

from langchain_core.documents import Document

from graph.build import (
    MAX_RETRIES,
    build_graph_v2,
    make_grade_documents,
    route_after_grade,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, _prompt):
        return SimpleNamespace(content=self.responses.pop(0))


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs
        self.search_calls = 0

    def similarity_search(self, _query, k=4):
        self.search_calls += 1
        return self.docs


def test_route_after_grade_relevant_goes_to_generate():
    state = {"documents": [Document(page_content="x")], "retries": 0}
    assert route_after_grade(state) == "generate"


def test_route_after_grade_irrelevant_goes_to_rewrite():
    state = {"documents": [], "retries": 0}
    assert route_after_grade(state) == "rewrite_query"


def test_route_after_grade_stops_after_max_retries():
    state = {"documents": [], "retries": MAX_RETRIES}
    assert route_after_grade(state) == "generate"


def test_grade_documents_marks_irrelevant_as_empty():
    grade = make_grade_documents(FakeLLM(["no"]))
    state = {"question": "q", "documents": [Document(page_content="irrelevant text")]}
    result = grade(state)
    assert result["documents"] == []


def test_graph_v2_falls_back_after_max_retries_without_infinite_loop():
    vectorstore = FakeVectorStore(docs=[Document(page_content="irrelevant")])
    llm = FakeLLM(responses=["no", "reformulated question 1", "no", "reformulated question 2", "no"])
    graph = build_graph_v2(llm, vectorstore)

    result = graph.invoke({"question": "what is x", "documents": [], "generation": "", "retries": 0})

    assert result["generation"] == "Je n'ai pas assez d'information dans la documentation indexée pour répondre."
    assert vectorstore.search_calls == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_graph_routing.py -v`
Expected: `ImportError` / `ModuleNotFoundError` — `MAX_RETRIES`, `build_graph_v2`, `make_grade_documents`, `route_after_grade` don't exist yet.

- [ ] **Step 3: Add v2 nodes to `graph/build.py`**

Append to the existing `graph/build.py` from Task 4:

```python
MAX_RETRIES = 2


def make_grade_documents(llm):
    def grade_documents(state: RAGState) -> dict:
        docs_text = "\n\n".join(doc.page_content for doc in state["documents"])
        response = llm.invoke(
            "Answer strictly 'yes' or 'no'. Does the context below contain enough "
            "information to answer the question?\n\n"
            f"Question: {state['question']}\n\nContext:\n{docs_text}"
        )
        relevant = "yes" in response.content.strip().lower()
        return {"documents": state["documents"] if relevant else []}

    return grade_documents


def route_after_grade(state: RAGState) -> str:
    if state["documents"]:
        return "generate"
    if state["retries"] >= MAX_RETRIES:
        return "generate"
    return "rewrite_query"


def make_rewrite_query(llm):
    def rewrite_query(state: RAGState) -> dict:
        response = llm.invoke(
            "Reformulate this question to improve document retrieval. "
            f"Keep it concise, return only the reformulated question:\n{state['question']}"
        )
        return {"question": response.content.strip(), "retries": state["retries"] + 1}

    return rewrite_query


def build_graph_v2(llm, vectorstore):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve(vectorstore))
    graph.add_node("grade_documents", make_grade_documents(llm))
    graph.add_node("rewrite_query", make_rewrite_query(llm))
    graph.add_node("generate", make_generate(llm))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_graph_routing.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add graph/build.py tests/
git commit -m "feat: add v2 agentic graph with grade-rewrite loop"
```

---

## Task 6: FastAPI service

**Files:**
- Create: `api/__init__.py` (empty)
- Create: `api/main.py`

**Interfaces:**
- Consumes: `config.get_llm()`, `config.get_vectorstore()`, `config.QDRANT_URL`, `graph.build.build_graph_v2`.
- Produces: FastAPI `app` with `GET /health -> {"status": "ok"}` (or `502` on Qdrant failure) and `POST /query` accepting `{"question": str}`, returning `{"answer": str, "sources": list[str]}`. Task 7's UI depends on this exact request/response shape.

- [ ] **Step 1: Write `api/__init__.py`**

Empty file.

- [ ] **Step 2: Write `api/main.py`**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

from config import QDRANT_URL, get_llm, get_vectorstore
from graph.build import build_graph_v2

app = FastAPI(title="rag-lab")
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph_v2(get_llm(), get_vectorstore())
    return _graph


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    try:
        QdrantClient(url=QDRANT_URL).get_collections()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant unreachable: {exc}") from exc
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        result = get_graph().invoke(
            {"question": request.question, "documents": [], "generation": "", "retries": 0}
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    sources = sorted({doc.metadata.get("path", "") for doc in result["documents"]})
    return QueryResponse(answer=result["generation"], sources=sources)
```

- [ ] **Step 3: Verify manually**

Run: `.venv/bin/uvicorn api.main:app --port 8000 &`

Run: `curl -s http://localhost:8000/health`
Expected: `{"status":"ok"}`

Run: `curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "What is a StateGraph?"}'`
Expected: JSON with a non-empty `answer` and a `sources` array of doc paths.

Then: `kill %1` to stop the dev server.

- [ ] **Step 4: Commit**

```bash
git add api/
git commit -m "feat: add FastAPI query and health endpoints"
```

---

## Task 7: Gradio chat UI

**Files:**
- Create: `ui/__init__.py` (empty)
- Create: `ui/app.py`

**Interfaces:**
- Consumes: the API contract from Task 6 (`POST {RAG_API_URL}/query` with `{"question": str}` → `{"answer": str, "sources": list[str]}`).
- Produces: a runnable Gradio app (`python -m ui.app`) — no further tasks depend on this one.

- [ ] **Step 1: Write `ui/__init__.py`**

Empty file.

- [ ] **Step 2: Write `ui/app.py`**

```python
import os

import gradio as gr
import httpx

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def ask(message: str, _history) -> str:
    response = httpx.post(f"{API_URL}/query", json={"question": message}, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not data["sources"]:
        return data["answer"]
    sources = "\n".join(f"- {s}" for s in data["sources"])
    return f"{data['answer']}\n\nSources:\n{sources}"


demo = gr.ChatInterface(fn=ask, title="rag-lab — LangChain/LangGraph doc assistant")

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 3: Verify manually**

Prerequisite: the FastAPI service from Task 6 running on port 8000.

Run: `.venv/bin/python -m ui.app`
Expected: prints a local URL (e.g. `http://127.0.0.1:7860`). Open it, ask "What is a StateGraph in LangGraph?", confirm a grounded answer with sources appears in the chat.

- [ ] **Step 4: Commit**

```bash
git add ui/
git commit -m "feat: add Gradio chat UI"
```
