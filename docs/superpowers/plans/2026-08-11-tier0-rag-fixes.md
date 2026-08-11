# Tier 0 RAG Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three already-identified weak spots in the RAG pipeline before layering any new retrieval technique on top: ingestion duplicates points on every rerun, document grading is all-or-nothing instead of per-document, and there's no tracing into what the graph actually does at each step.

**Architecture:** No new components. Three surgical, independent changes to existing files: `ingestion/ingest.py` (recreate collection instead of appending), `graph/build.py` (grade each retrieved `Document` individually instead of the concatenated batch), and `.env.example`/`README.md` (wire up LangSmith tracing via environment variables — LangChain auto-detects these, no code change needed).

**Tech Stack:** Python 3.11+, LangChain/LangGraph, `langchain-qdrant`, `pytest`. No new dependencies.

## Global Constraints

- Type hints on all function signatures and return types (per `~/.claude/rules/python.md`)
- Prefer f-strings over `.format()`/`%`
- Import order: stdlib → third-party → local, alphabetical within each group
- Never use bare `except:` — always specify exception type
- Comments in English (per project `CLAUDE.md`)
- Commits: English, imperative mood, conventional-commits prefix (`fix:`/`feat:`), max 72-char subject, no trailing period (per `~/.claude/rules/git.md`)
- Run tests with `.venv/bin/pytest tests/ -v` (matches `.github/workflows/ci.yml`)
- Trivial config/one-line changes don't need a dedicated automated test (ponytail rule already in effect on this session) — use the manual verification steps given instead

---

### Task 1: Idempotent ingestion (recreate collection instead of appending)

**Files:**
- Modify: `ingestion/ingest.py:86-99`

**Interfaces:**
- Consumes: nothing new — `QdrantVectorStore.from_texts(..., force_recreate: bool = False)` is an existing parameter of the installed `langchain-qdrant` version (verified via `inspect.signature`); when `True`, it deletes and recreates the collection before adding points instead of appending to it.
- Produces: nothing consumed elsewhere — this is a leaf change.

**Context:** `main()` already does a full recompute every run (fresh `git clone`, re-chunks and re-embeds every source), so there is no incremental-ingestion use case to preserve. The correct fix is "replace the collection", not "assign deterministic IDs and upsert" — the latter would be more code for a behavior (partial/incremental ingestion) this script doesn't have and doesn't need (YAGNI).

- [ ] **Step 1: Replace the `from_texts` call**

In `ingestion/ingest.py`, replace the comment and call (lines 86-99):

```python
    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        force_recreate=True,
    )
```

Delete the now-stale comment above it (`# ponytail: from_texts() appends rather than upserting...`) — the behavior it warned about no longer exists.

- [ ] **Step 2: Update the module docstring/log line if needed**

The existing `print(f"Ingested {len(texts)} chunks into '{QDRANT_COLLECTION}'.")` at the end of `main()` stays as-is — still accurate.

- [ ] **Step 3: Manual verification (no live Qdrant in CI, so this is not an automated test)**

```bash
docker compose up -d qdrant
.venv/bin/python -m ingestion.ingest
.venv/bin/python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='http://localhost:6333')
print('after run 1:', c.count(collection_name='langchain_docs').count)
"
.venv/bin/python -m ingestion.ingest
.venv/bin/python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='http://localhost:6333')
print('after run 2:', c.count(collection_name='langchain_docs').count)
"
```

Expected: the two counts are identical (previously, the second would have been double the first).

- [ ] **Step 4: Commit**

```bash
git add ingestion/ingest.py
git commit -m "fix: recreate Qdrant collection on ingestion instead of appending"
```

---

### Task 2: Per-document grading in `grade_documents`

**Files:**
- Modify: `graph/build.py:44-56`
- Test: `tests/test_graph_routing.py`

**Interfaces:**
- Consumes: `RAGState` (`graph/state.py`) — unchanged shape: `{question: str, documents: list[Document], generation: str, retries: int}`.
- Produces: `grade_documents(state) -> {"documents": list[Document]}` — same return shape as before (a dict with a `documents` key), so `route_after_grade` and the graph wiring in `build_graph_v2` need no changes. Only the *content* of that list changes: previously all-or-nothing, now a genuine per-document filter.

**Context:** Today `grade_documents` concatenates every retrieved document into one blob and asks the LLM one yes/no question about the whole thing — so 1 relevant doc + 3 irrelevant ones either keeps all 4 or drops all 4. Grading each document independently is the actual Corrective-RAG pattern and lets partially-relevant retrievals still contribute their good document(s).

- [ ] **Step 1: Write the failing test — partial relevance is preserved**

Add to `tests/test_graph_routing.py` (after `test_grade_documents_marks_irrelevant_as_empty`):

```python
def test_grade_documents_keeps_only_relevant_ones():
    grade = make_grade_documents(FakeLLM(["yes", "no"]))
    relevant_doc = Document(page_content="relevant text")
    irrelevant_doc = Document(page_content="irrelevant text")
    state = {"question": "q", "documents": [relevant_doc, irrelevant_doc]}
    result = grade(state)
    assert result["documents"] == [relevant_doc]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_graph_routing.py::test_grade_documents_keeps_only_relevant_ones -v`
Expected: FAIL — current implementation grades the concatenation of both documents with a single LLM call (only consumes one of the two queued `FakeLLM` responses) and returns both-or-neither, not `[relevant_doc]`.

- [ ] **Step 3: Implement per-document grading**

Replace `make_grade_documents` in `graph/build.py:44-56` with:

```python
def make_grade_documents(llm):
    def is_relevant(question: str, doc: Document) -> bool:
        response = llm.invoke(
            "Answer strictly 'yes' or 'no'. Does the context below contain "
            "information relevant to the question, even partially? Answer "
            "'yes' unless the context is completely unrelated to the topic.\n\n"
            f"Question: {question}\n\nContext:\n{doc.page_content}"
        )
        return "yes" in response.content.strip().lower()

    def grade_documents(state: RAGState) -> dict:
        relevant_docs = [doc for doc in state["documents"] if is_relevant(state["question"], doc)]
        return {"documents": relevant_docs}

    return grade_documents
```

Add the import at the top of `graph/build.py` (third-party group, alphabetical):

```python
from langchain_core.documents import Document
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_graph_routing.py::test_grade_documents_keeps_only_relevant_ones -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `.venv/bin/pytest tests/ -v`
Expected: all tests pass, including `test_grade_documents_marks_irrelevant_as_empty` (single doc, single "no" response — still correct under per-document grading) and `test_graph_v2_falls_back_after_max_retries_without_infinite_loop` (the fake vectorstore always returns exactly one document per retrieval, so it issues exactly one grading call per retry, same as before).

- [ ] **Step 6: Commit**

```bash
git add graph/build.py tests/test_graph_routing.py
git commit -m "fix: grade retrieved documents individually instead of as one batch"
```

---

### Task 3: LangSmith tracing

**Files:**
- Modify: `.env.example`
- Modify: `README.md` (Configuration section, table around line 65-74)

**Interfaces:**
- Consumes: nothing — LangChain reads `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` directly from the environment; `config.py` already calls `load_dotenv()` before anything else runs, and `docker-compose.yml`'s `api`/`ingestion`/`ui` services already load `.env` via `env_file:`, so no code or compose changes are needed.
- Produces: nothing consumed elsewhere.

**Context:** This project is a learning lab specifically about how the LangGraph pipeline behaves — tracing every node's input/output is high-value here and costs zero code changes, just environment variables to document.

- [ ] **Step 1: Add tracing variables to `.env.example`**

Append to `.env.example`:

```bash
# To enable LangSmith tracing (see every graph node's input/output at smith.langchain.com):
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=lsv2_...
# LANGCHAIN_PROJECT=rag-lab
```

- [ ] **Step 2: Document it in `README.md`**

In the "Configuration" section, after the existing variables table and its API-provider paragraph, add:

```markdown
Pour tracer chaque étape du graphe (utile pour comprendre `retrieve → grade → rewrite → generate`) : décommenter `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY` et `LANGCHAIN_PROJECT` dans `.env` — LangChain envoie alors les traces à [smith.langchain.com](https://smith.langchain.com) sans aucun changement de code.
```

- [ ] **Step 3: Manual verification**

```bash
cp .env.example .env
# edit .env: uncomment the three LANGCHAIN_* lines, set a real LANGCHAIN_API_KEY
docker compose up -d qdrant api ui
curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"question": "what is a StateGraph?"}'
```

Expected: a new trace appears under the `rag-lab` project at smith.langchain.com, showing the `retrieve`/`grade_documents`/`rewrite_query`/`generate` nodes.

- [ ] **Step 4: Commit**

```bash
git add .env.example README.md
git commit -m "docs: document optional LangSmith tracing setup"
```

---

## Self-Review

- **Spec coverage:** all three Tier 0 items from `docs/superpowers/specs/2026-08-11-advanced-rag-improvements-design.md` are covered — idempotent ingestion (Task 1), per-document grading (Task 2), LangSmith tracing (Task 3).
- **Placeholders:** none — every step has literal code/commands.
- **Type consistency:** `grade_documents` keeps its existing signature and return shape (`dict` with `documents` key); `route_after_grade` and `build_graph_v2` wiring are untouched.
- **Scope:** each task is independently committable and independently valuable; no task depends on another.
