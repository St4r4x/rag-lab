# Tier 1 Hybrid Search + Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hybrid (dense + sparse) retrieval and cross-encoder reranking to the RAG pipeline, so `retrieve` pulls a wider candidate pool via Qdrant's native RRF fusion and `rerank` trims it back down to the 4 documents `grade_documents`/`generate` already expect.

**Architecture:** One new dependency (`fastembed`) covers both new capabilities. `config.py` gains `get_sparse_embeddings()` and `get_reranker()`, and `get_vectorstore()` switches to hybrid mode. `ingestion/ingest.py` indexes both dense and sparse vectors. `graph/build.py` gains a `rerank` node wired into `build_graph_v2` only (`build_graph_v1`, the deliberately-naive baseline, is untouched). `api/main.py` wires the reranker into `get_graph()`.

**Tech Stack:** Python 3.11+, LangChain/LangGraph, `langchain-qdrant`, `fastembed` (new), `pytest`.

## Global Constraints

- Type hints on all function signatures and return types (per `~/.claude/rules/python.md`)
- Prefer f-strings over `.format()`/`%`
- Import order: stdlib → third-party → local, alphabetical within each group (verify placement — Tier 0 shipped one deferred cosmetic miss here, don't repeat it)
- Never use bare `except:` — always specify exception type
- Comments in English (per project `CLAUDE.md`)
- Commits: English, imperative mood, conventional-commits prefix (`fix:`/`feat:`), max 72-char subject, no trailing period (per `~/.claude/rules/git.md`)
- Run tests with `.venv/bin/pytest tests/ -v` (matches `.github/workflows/ci.yml`)
- `build_graph_v1` and its call sites must not change behavior — it's the intentionally-simple baseline this project keeps for comparison (see README)
- Trivial config/wiring changes that mirror an existing untested pattern in the same file (e.g. `get_llm()`/`get_embeddings()` have no dedicated tests) don't need a new dedicated automated test — this plan says explicitly, per task, where that applies

---

### Task 1: Config — sparse embeddings, hybrid vectorstore, reranker factory

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `get_sparse_embeddings() -> FastEmbedSparse` and `get_reranker() -> TextCrossEncoder` — Task 2 (ingestion) calls `get_sparse_embeddings()`; Task 3 (graph/API wiring) calls `get_reranker()`. `get_vectorstore()`'s return type is unchanged (`QdrantVectorStore`), only its internal construction changes.

**Context:** `config.py` currently has `get_llm()`, `get_embeddings()`, and `get_vectorstore()` (`lru_cache`d), each a thin factory with no dedicated test — this task follows that exact pattern for the two new factories.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to the `dependencies` list (alphabetically, after `"fastapi>=0.115.0"` and before `"gradio>=5.0.0"`... check exact alphabetical order against the full list — `fastembed` sorts after `fastapi` and before `gradio`):

```toml
    "fastembed>=0.8.0",
```

- [ ] **Step 2: Install and verify import**

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; from langchain_qdrant import FastEmbedSparse, RetrievalMode; print('ok')"
```

Expected: prints `ok` with no import errors.

- [ ] **Step 3: Add the two new factories and update `get_vectorstore()`**

In `config.py`, add these imports (third-party group, alphabetical — `fastembed` sorts before `langchain_qdrant`... no wait, check actual alphabetical order of the full merged import block):

```python
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
```

(`langchain_qdrant import QdrantVectorStore` already exists in this file — merge `FastEmbedSparse` and `RetrievalMode` into that same import statement, alphabetically: `FastEmbedSparse, QdrantVectorStore, RetrievalMode`.)

Add two new module-level constants next to the existing `QDRANT_URL`/`QDRANT_COLLECTION` env-var constants:

```python
SPARSE_EMBEDDING_MODEL = os.environ.get("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
```

Add `get_sparse_embeddings()` right after the existing `get_embeddings()`:

```python
def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL)
```

Add `get_reranker()` right after `get_vectorstore()`:

```python
@lru_cache
def get_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANKER_MODEL)
```

Update `get_vectorstore()` to construct in hybrid mode:

```python
@lru_cache
def get_vectorstore() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
    )
```

- [ ] **Step 4: Manual verification (no dedicated automated test — mirrors the existing untested `get_llm`/`get_embeddings` pattern in this file)**

```bash
.venv/bin/python -c "
from config import get_reranker, get_sparse_embeddings
sparse = get_sparse_embeddings()
print(type(sparse).__name__)
reranker = get_reranker()
scores = reranker.rerank('what is a StateGraph', ['a StateGraph is a graph', 'bananas are yellow'])
print(list(scores))
"
```

Expected: prints `FastEmbedSparse`, then a list of 2 floats where the first score is higher than the second (the first document is topically relevant to the query, the second isn't). This call downloads the reranker model (~80MB) on first run — requires network access once.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config.py
git commit -m "feat: add hybrid search and reranker config factories"
```

---

### Task 2: Ingestion — hybrid indexing

**Files:**
- Modify: `ingestion/ingest.py`

**Interfaces:**
- Consumes: `get_sparse_embeddings()` from `config.py` (Task 1).
- Produces: nothing consumed elsewhere — leaf change, same as Tier 0's Task 1.

**Context:** `main()` already calls `get_embeddings()` and passes `force_recreate=True` to `QdrantVectorStore.from_texts(...)` (Tier 0). This task adds the sparse side so the same rebuild also indexes sparse vectors.

- [ ] **Step 1: Update imports**

In `ingestion/ingest.py`, change:

```python
from langchain_qdrant import QdrantVectorStore
```

to:

```python
from langchain_qdrant import QdrantVectorStore, RetrievalMode
```

and change:

```python
from config import QDRANT_COLLECTION, QDRANT_URL, get_embeddings
```

to:

```python
from config import QDRANT_COLLECTION, QDRANT_URL, get_embeddings, get_sparse_embeddings
```

- [ ] **Step 2: Construct sparse embeddings and pass them to `from_texts`**

In `main()`, change:

```python
    embeddings = get_embeddings()
```

to:

```python
    embeddings = get_embeddings()
    sparse_embeddings = get_sparse_embeddings()
```

And change the `from_texts(...)` call to:

```python
    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        force_recreate=True,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
    )
```

- [ ] **Step 3: Manual verification (needs live Qdrant + embeddings backend, same constraint as Tier 0's Task 1)**

```bash
docker compose up -d qdrant
.venv/bin/python -m ingestion.ingest
.venv/bin/python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='http://localhost:6333')
info = c.get_collection('langchain_docs')
print('vectors config:', info.config.params.vectors)
print('sparse vectors config:', info.config.params.sparse_vectors)
"
```

Expected: `sparse vectors config` is non-empty/non-`None` (previously it would have been `None` before this task). If Ollama isn't reachable in this environment (same constraint documented in Tier 0's plan), fall back to a static-code check confirming the diff matches Steps 1-2 exactly, and report which verification you actually performed — do not claim the runtime check passed if it didn't run.

- [ ] **Step 4: Commit**

```bash
git add ingestion/ingest.py
git commit -m "feat: index sparse vectors alongside dense at ingestion"
```

---

### Task 3: Graph — rerank node, retrieve pool size, API wiring

**Files:**
- Modify: `graph/build.py`
- Modify: `api/main.py`
- Test: `tests/test_graph_routing.py`

**Interfaces:**
- Consumes: `get_reranker()` from `config.py` (Task 1) — used in `api/main.py` only, not in `graph/build.py` itself (the graph module stays free of config/import-time side effects, consistent with `llm`/`vectorstore` already being passed in rather than constructed inside `graph/build.py`).
- Produces: `build_graph_v2(llm, vectorstore, reranker)` — signature gains a required third positional parameter. `build_graph_v1(llm, vectorstore)` is unchanged. `make_retrieve(vectorstore, k=4)` gains an optional `k` parameter, default preserves current behavior.

**Context:** This is the task that actually wires reranking into the graph. `make_retrieve`'s new `k` parameter lets `build_graph_v2` request a bigger candidate pool (20) while `build_graph_v1` keeps the original 4 by omitting the argument. The new `rerank` node sits between `retrieve` and `grade_documents` in `build_graph_v2`'s wiring only.

- [ ] **Step 1: Write the failing tests**

In `tests/test_graph_routing.py`, update the import block (add `make_rerank`, alphabetically between `make_grade_documents` and `route_after_grade`):

```python
from graph.build import (
    MAX_RETRIES,
    build_graph_v2,
    make_grade_documents,
    make_rerank,
    route_after_grade,
)
```

Add a `FakeReranker` class after `FakeVectorStore`:

```python
class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def rerank(self, _query, _documents):
        return self.scores
```

Add two new tests after `test_grade_documents_keeps_only_relevant_ones`:

```python
def test_rerank_sorts_by_score_and_keeps_top_k():
    docs = [Document(page_content=f"doc{i}") for i in range(5)]
    reranker = FakeReranker(scores=[0.1, 0.9, 0.5, 0.8, 0.2])
    rerank = make_rerank(reranker)
    state = {"question": "q", "documents": docs}
    result = rerank(state)
    assert result["documents"] == [docs[1], docs[3], docs[2], docs[4]]


def test_rerank_handles_empty_documents():
    rerank = make_rerank(FakeReranker(scores=[]))
    state = {"question": "q", "documents": []}
    result = rerank(state)
    assert result["documents"] == []
```

Update `test_graph_v2_falls_back_after_max_retries_without_infinite_loop` to pass a reranker (`build_graph_v2` will require a third argument once Step 3 lands):

```python
def test_graph_v2_falls_back_after_max_retries_without_infinite_loop():
    vectorstore = FakeVectorStore(docs=[Document(page_content="irrelevant")])
    reranker = FakeReranker(scores=[1.0])
    llm = FakeLLM(responses=["no", "reformulated question 1", "no", "reformulated question 2", "no"])
    graph = build_graph_v2(llm, vectorstore, reranker)

    result = graph.invoke({"question": "what is x", "documents": [], "generation": "", "retries": 0})

    assert result["generation"] == "Je n'ai pas assez d'information dans la documentation indexée pour répondre."
    assert vectorstore.search_calls == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_graph_routing.py -v`
Expected: `test_rerank_sorts_by_score_and_keeps_top_k` and `test_rerank_handles_empty_documents` FAIL with `ImportError`/`AttributeError` (`make_rerank` doesn't exist yet). `test_graph_v2_falls_back_after_max_retries_without_infinite_loop` FAILS with `TypeError: build_graph_v2() takes 2 positional arguments but 3 were given` (or similar — the function doesn't accept a third argument yet).

- [ ] **Step 3: Implement `make_rerank`, parameterize `make_retrieve`, wire `build_graph_v2`**

In `graph/build.py`, change `make_retrieve`:

```python
def make_retrieve(vectorstore, k=4):
    def retrieve(state: RAGState) -> dict:
        docs = vectorstore.similarity_search(state["question"], k=k)
        return {"documents": docs}

    return retrieve
```

Leave `build_graph_v1` exactly as-is (`make_retrieve(vectorstore)` still uses the default `k=4`).

Add two new constants next to `MAX_RETRIES`:

```python
MAX_RETRIES = 2
RETRIEVE_POOL_SIZE = 20
RERANK_TOP_K = 4
```

Add `make_rerank` after `make_grade_documents`:

```python
def make_rerank(reranker):
    def rerank(state: RAGState) -> dict:
        if not state["documents"]:
            return {"documents": []}
        scores = reranker.rerank(
            state["question"], [doc.page_content for doc in state["documents"]]
        )
        ranked = sorted(zip(scores, state["documents"]), key=lambda pair: pair[0], reverse=True)
        return {"documents": [doc for _, doc in ranked][:RERANK_TOP_K]}

    return rerank
```

Update `build_graph_v2`:

```python
def build_graph_v2(llm, vectorstore, reranker):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve(vectorstore, k=RETRIEVE_POOL_SIZE))
    graph.add_node("rerank", make_rerank(reranker))
    graph.add_node("grade_documents", make_grade_documents(llm))
    graph.add_node("rewrite_query", make_rewrite_query(llm))
    graph.add_node("generate", make_generate(llm))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()
```

- [ ] **Step 4: Wire the reranker into the API**

In `api/main.py`, change:

```python
from config import QDRANT_URL, get_llm, get_vectorstore
```

to:

```python
from config import QDRANT_URL, get_llm, get_reranker, get_vectorstore
```

and change:

```python
@lru_cache
def get_graph():
    return build_graph_v2(get_llm(), get_vectorstore())
```

to:

```python
@lru_cache
def get_graph():
    return build_graph_v2(get_llm(), get_vectorstore(), get_reranker())
```

No dedicated test for this one-line wiring change — `api/main.py` has no existing test coverage for `get_graph()` today (same untested-wiring precedent as Task 1's factories).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_graph_routing.py -v`
Expected: all tests PASS, including the two new ones and the updated fallback test.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all tests pass, pristine output.

- [ ] **Step 7: Commit**

```bash
git add graph/build.py api/main.py tests/test_graph_routing.py
git commit -m "feat: add reranking node and widen retrieval pool for hybrid search"
```

---

## Self-Review

- **Spec coverage:** both Tier 1 items from `docs/superpowers/specs/2026-08-11-tier1-hybrid-search-reranking-design.md` are covered — hybrid search (Tasks 1-2) and reranking (Tasks 1, 3). Contextual retrieval is explicitly out of scope per the spec and this plan.
- **Placeholders:** none — every step has literal code/commands, including exact expected outputs.
- **Type consistency:** `make_retrieve(vectorstore, k=4) -> retrieve`, `make_rerank(reranker) -> rerank`, both matching the `state: RAGState -> dict` shape used by every other node in the file. `build_graph_v2(llm, vectorstore, reranker)`'s new third parameter is threaded consistently through Task 3's graph wiring, its test, and `api/main.py`'s call site — checked all three use the same order (`llm, vectorstore, reranker`).
- **Scope:** Task 1 has no dependents within itself; Tasks 2 and 3 each depend only on Task 1 (not on each other) — could be reviewed/executed in either order after Task 1, but this plan sequences them 1→2→3 for a single coherent worktree history.
- **`build_graph_v1` regression check:** confirmed no step touches `build_graph_v1` or changes `make_retrieve`'s default (`k=4`), so the naive baseline graph's behavior is unchanged.
