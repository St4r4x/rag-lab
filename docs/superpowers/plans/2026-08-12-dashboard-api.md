# Dashboard API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the API endpoints a future Gradio dashboard (Sub-project B, separate plan) will consume: read-only config, evaluation run listing/detail/trigger, and additive document upload — without touching the existing `/health`/`/query` behavior or the chat UI.

**Architecture:** `api/main.py` splits into `api/dependencies.py` (the shared `get_graph()` factory, extracted to avoid a circular import), `api/eval_routes.py` (an `APIRouter` for the three eval endpoints), and `api/documents_routes.py` (an `APIRouter` for upload). `eval/run_eval.py` gains a pure `summarize()` used by both the CLI script and the API. `ingestion/ingest.py` gains a pure `chunk_document()` used by both the standalone ingestion script and the new upload endpoint.

**Tech Stack:** FastAPI, Pydantic, `fastapi.testclient.TestClient` (uses the already-installed `httpx`) for new tests. No new dependencies.

## Global Constraints

- Type hints on all function signatures and return types (per `~/.claude/rules/python.md`)
- Prefer f-strings over `.format()`/`%`
- Use `pathlib.Path` instead of `os.path`
- Import order: stdlib → third-party → local, alphabetical within each group
- Never use bare `except:` — always specify exception type
- Comments in English (per project `CLAUDE.md`)
- Commits: English, imperative mood, conventional-commits prefix (`feat:`/`refactor:`), max 72-char subject, no trailing period (per `~/.claude/rules/git.md`)
- Run tests with `.venv/bin/pytest tests/ -v` (matches `.github/workflows/ci.yml`)
- **Lesson from every prior plan on this project:** when editing an existing list/file (imports, dependencies, docker-compose services), add without deleting or reordering existing content unless explicitly told to. Re-read your diff line by line before committing.
- No dedicated automated test for endpoints that need a live Qdrant/LLM backend (`POST /eval/run`, `POST /documents`) — same precedent as `ingestion/ingest.py` and `eval/run_eval.py`. Manual verification only, and say explicitly which kind of verification you performed.
- `run_id` path parameters must be validated against `^\d{8}T\d{6}Z$` *before* any filesystem access — this is a security-relevant branch, not a formatting nicety.

---

### Task 1: `api/dependencies.py`, `GET /config`, `summarize()` extraction, pytest config

**Files:**
- Create: `api/dependencies.py`
- Modify: `api/main.py`
- Modify: `eval/run_eval.py`
- Modify: `pyproject.toml`
- Test: `tests/test_run_eval.py` (new)
- Test: `tests/test_api.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `api.dependencies.get_graph()` (Task 3 will import this into `api/eval_routes.py`). `eval.run_eval.summarize(results: list[dict]) -> dict` returning `{"count": int, "avg_faithfulness": float | None, "faithfulness_scored": int, "avg_correctness": float | None, "correctness_scored": int}` (Tasks 2 and 3 both consume this).

**Context:** `api/main.py` currently defines `get_graph()` inline. Extracting it into its own module now means `api/eval_routes.py` (Task 3) can import it without creating a circular import (`main.py` → `eval_routes.py` → `main.py`). `eval/run_eval.py`'s `print_summary()` currently computes averages inline — extracting that computation into `summarize()` lets the API report the same numbers without duplicating the logic.

- [ ] **Step 1: Create `api/dependencies.py`**

```python
# api/dependencies.py
from functools import lru_cache

from config import get_llm, get_reranker, get_vectorstore
from graph.build import build_graph_v2


@lru_cache
def get_graph():
    return build_graph_v2(get_llm(), get_vectorstore(), get_reranker())
```

- [ ] **Step 2: Rewrite `api/main.py`**

Replace the entire file with:

```python
# api/main.py
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

from api.dependencies import get_graph
from config import QDRANT_COLLECTION, QDRANT_URL

app = FastAPI(title="rag-lab")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class ConfigResponse(BaseModel):
    llm_model: str
    embedding_model: str
    sparse_embedding_model: str
    reranker_model: str
    judge_model: str
    qdrant_url: str
    qdrant_collection: str


@app.get("/health")
def health():
    try:
        QdrantClient(url=QDRANT_URL).get_collections()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant unreachable: {exc}") from exc
    return {"status": "ok"}


@app.get("/config", response_model=ConfigResponse)
def get_config():
    llm_model = os.environ.get("LLM_MODEL", "ollama:llama3.2:3b")
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "") or llm_model
    return ConfigResponse(
        llm_model=llm_model,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "ollama:nomic-embed-text"),
        sparse_embedding_model=os.environ.get("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25"),
        reranker_model=os.environ.get("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        judge_model=judge_model,
        qdrant_url=QDRANT_URL,
        qdrant_collection=QDRANT_COLLECTION,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        result = get_graph().invoke(
            {"question": request.question, "documents": [], "generation": "", "retries": 0}
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    sources = sorted({doc.metadata.get("url", "") for doc in result["documents"]})
    return QueryResponse(answer=result["generation"], sources=sources)
```

(This removes the inline `get_graph()`/`@lru_cache`/`build_graph_v2` import — moved to `api/dependencies.py` — and adds `os`, `QDRANT_COLLECTION`, `ConfigResponse`, and the `/config` route. `/health` and `/query` behavior is unchanged.)

- [ ] **Step 3: Extract `summarize()` in `eval/run_eval.py`**

Replace the existing `print_summary` function with:

```python
def summarize(results: list[dict]) -> dict:
    faithfulness_scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    correctness_scores = [r["correctness"] for r in results if r["correctness"] is not None]
    return {
        "count": len(results),
        "avg_faithfulness": statistics.mean(faithfulness_scores) if faithfulness_scores else None,
        "faithfulness_scored": len(faithfulness_scores),
        "avg_correctness": statistics.mean(correctness_scores) if correctness_scores else None,
        "correctness_scored": len(correctness_scores),
    }


def print_summary(results: list[dict]) -> None:
    summary = summarize(results)
    print()
    if summary["avg_faithfulness"] is not None:
        print(
            f"Average faithfulness: {summary['avg_faithfulness']:.2f} "
            f"({summary['faithfulness_scored']}/{summary['count']} scored)"
        )
    if summary["avg_correctness"] is not None:
        print(
            f"Average correctness: {summary['avg_correctness']:.2f} "
            f"({summary['correctness_scored']}/{summary['count']} scored)"
        )
```

Everything else in `eval/run_eval.py` (imports, `load_dataset`, `evaluate_one`, `write_report`, `main`) is unchanged.

- [ ] **Step 4: Add pytest warning filter to `pyproject.toml`**

Append at the end of the file:

```toml

[tool.pytest.ini_options]
filterwarnings = [
    "ignore::starlette.exceptions.StarletteDeprecationWarning",
]
```

(`fastapi.testclient.TestClient` triggers this specific, non-actionable deprecation warning from the `starlette`/`httpx` ecosystem — unrelated to this project's code. This keeps test output pristine without suppressing anything else. Verified: with this filter in place, `pytest -W error::DeprecationWarning` still passes a `TestClient`-using test — the ignore rule takes precedence.)

- [ ] **Step 5: Write the tests**

Create `tests/test_run_eval.py`:

```python
# tests/test_run_eval.py
from eval.run_eval import summarize


def test_summarize_all_scored():
    results = [
        {"faithfulness": 4, "correctness": 5},
        {"faithfulness": 2, "correctness": 3},
    ]
    summary = summarize(results)
    assert summary == {
        "count": 2,
        "avg_faithfulness": 3.0,
        "faithfulness_scored": 2,
        "avg_correctness": 4.0,
        "correctness_scored": 2,
    }


def test_summarize_partially_scored():
    results = [
        {"faithfulness": 4, "correctness": None},
        {"faithfulness": None, "correctness": 3},
    ]
    summary = summarize(results)
    assert summary["count"] == 2
    assert summary["avg_faithfulness"] == 4.0
    assert summary["faithfulness_scored"] == 1
    assert summary["avg_correctness"] == 3.0
    assert summary["correctness_scored"] == 1


def test_summarize_none_scored():
    results = [{"faithfulness": None, "correctness": None}]
    summary = summarize(results)
    assert summary["count"] == 1
    assert summary["avg_faithfulness"] is None
    assert summary["faithfulness_scored"] == 0
    assert summary["avg_correctness"] is None
    assert summary["correctness_scored"] == 0
```

Create `tests/test_api.py`:

```python
# tests/test_api.py
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_get_config_returns_expected_fields():
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "llm_model",
        "embedding_model",
        "sparse_embedding_model",
        "reranker_model",
        "judge_model",
        "qdrant_url",
        "qdrant_collection",
    }


def test_get_config_never_exposes_secrets():
    response = client.get("/config")
    body = response.text.lower()
    assert "key" not in body
    assert "token" not in body
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: all tests pass (12 pre-existing + 5 new = 17), pristine output (no `StarletteDeprecationWarning`).

- [ ] **Step 7: Commit**

```bash
git add api/dependencies.py api/main.py eval/run_eval.py pyproject.toml tests/test_run_eval.py tests/test_api.py
git commit -m "feat: add GET /config and extract get_graph/summarize helpers"
```

---

### Task 2: `GET /eval/runs` and `GET /eval/runs/{run_id}`

**Files:**
- Create: `api/eval_routes.py`
- Modify: `api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `eval.run_eval.RESULTS_DIR`, `eval.run_eval.summarize()` (Task 1).
- Produces: `api.eval_routes.router` (an `APIRouter`) — Task 3 adds a third route to this same router; `api/main.py` mounts it via `app.include_router`.

**Context:** These two read-only endpoints list and inspect past evaluation runs. `run_id` is validated with a strict regex *before* it's used to build a filesystem path — this structurally rules out path traversal rather than checking for it after the fact.

- [ ] **Step 1: Create `api/eval_routes.py`**

```python
# api/eval_routes.py
import json
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from eval.run_eval import RESULTS_DIR, summarize

router = APIRouter(prefix="/eval", tags=["eval"])

RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")


class EvalRunSummary(BaseModel):
    id: str
    count: int
    avg_faithfulness: float | None
    avg_correctness: float | None


def _summary_response(run_id: str, results: list[dict]) -> EvalRunSummary:
    summary = summarize(results)
    return EvalRunSummary(
        id=run_id,
        count=summary["count"],
        avg_faithfulness=summary["avg_faithfulness"],
        avg_correctness=summary["avg_correctness"],
    )


@router.get("/runs", response_model=list[EvalRunSummary])
def list_eval_runs():
    if not RESULTS_DIR.exists():
        return []
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        results = json.loads(path.read_text(encoding="utf-8"))
        runs.append(_summary_response(path.stem, results))
    return runs


@router.get("/runs/{run_id}")
def get_eval_run(run_id: str):
    if not RUN_ID_PATTERN.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run id")
    path = RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Mount the router in `api/main.py`**

Change:

```python
from api.dependencies import get_graph
from config import QDRANT_COLLECTION, QDRANT_URL
```

to:

```python
from api import eval_routes
from api.dependencies import get_graph
from config import QDRANT_COLLECTION, QDRANT_URL
```

Change:

```python
app = FastAPI(title="rag-lab")
```

to:

```python
app = FastAPI(title="rag-lab")
app.include_router(eval_routes.router)
```

- [ ] **Step 3: Write the tests**

Add to `tests/test_api.py`:

```python
import json

from eval.run_eval import RESULTS_DIR as _  # noqa: F401  (imported for monkeypatch target discovery)


def test_list_eval_runs_empty_when_no_results_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("api.eval_routes.RESULTS_DIR", tmp_path / "missing")
    response = client.get("/eval/runs")
    assert response.status_code == 200
    assert response.json() == []


def test_list_eval_runs_returns_summaries(monkeypatch, tmp_path):
    (tmp_path / "20260101T000000Z.json").write_text(
        json.dumps([{"faithfulness": 4, "correctness": 5}]), encoding="utf-8"
    )
    monkeypatch.setattr("api.eval_routes.RESULTS_DIR", tmp_path)
    response = client.get("/eval/runs")
    assert response.status_code == 200
    assert response.json() == [
        {"id": "20260101T000000Z", "count": 1, "avg_faithfulness": 4.0, "avg_correctness": 5.0}
    ]


def test_get_eval_run_rejects_malformed_run_id():
    response = client.get("/eval/runs/not-a-valid-id")
    assert response.status_code == 400


def test_get_eval_run_returns_404_for_missing_well_formed_id(monkeypatch, tmp_path):
    monkeypatch.setattr("api.eval_routes.RESULTS_DIR", tmp_path)
    response = client.get("/eval/runs/20260101T000000Z")
    assert response.status_code == 404


def test_get_eval_run_returns_full_detail(monkeypatch, tmp_path):
    detail = [{"id": "q01", "faithfulness": 4, "correctness": 5}]
    (tmp_path / "20260101T000000Z.json").write_text(json.dumps(detail), encoding="utf-8")
    monkeypatch.setattr("api.eval_routes.RESULTS_DIR", tmp_path)
    response = client.get("/eval/runs/20260101T000000Z")
    assert response.status_code == 200
    assert response.json() == detail
```

Note on the `test_get_eval_run_rejects_malformed_run_id` test: a multi-segment traversal string like `/eval/runs/../../etc/passwd` never reaches our handler at all — Starlette's router itself resolves/rejects it as 404 before our code runs (verified empirically while writing this plan). That's a real, structural guarantee from the framework, but it means testing *our* validation branch specifically requires a single-path-segment string that reaches the handler and fails the regex — `not-a-valid-id` is exactly that.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: all tests pass (17 pre-existing + 5 new = 22), pristine output.

- [ ] **Step 5: Commit**

```bash
git add api/eval_routes.py api/main.py tests/test_api.py
git commit -m "feat: add eval run listing and detail endpoints"
```

---

### Task 3: `POST /eval/run` + docker-compose volume for `api`

**Files:**
- Modify: `api/eval_routes.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `api.dependencies.get_graph()` (Task 1), `config.get_judge_llm()` (existing), `eval.run_eval.load_dataset`/`evaluate_one`/`write_report` (existing), `_summary_response` (Task 2, same file).
- Produces: nothing new consumed elsewhere.

**Context:** Triggers a fresh evaluation run and returns its summary immediately (this is a synchronous, potentially slow — tens of seconds to a few minutes with a local model — HTTP call; Sub-project B's UI will need a loading indicator, not this task's concern). The `api` container needs the same `eval/results/` bind mount `eval` already has, or writes from this endpoint would be lost on container restart exactly like Tier 2's already-fixed bug.

- [ ] **Step 1: Add the imports and endpoint to `api/eval_routes.py`**

Change:

```python
from eval.run_eval import RESULTS_DIR, summarize
```

to:

```python
from api.dependencies import get_graph
from config import get_judge_llm
from eval.run_eval import RESULTS_DIR, evaluate_one, load_dataset, summarize, write_report
```

Add at the end of the file:

```python
@router.post("/run", response_model=EvalRunSummary)
def run_eval():
    judge_llm = get_judge_llm()
    graph = get_graph()
    results = [evaluate_one(graph, judge_llm, item) for item in load_dataset()]
    out_path = write_report(results)
    return _summary_response(out_path.stem, results)
```

- [ ] **Step 2: Add the volume mount to the `api` service in `docker-compose.yml`**

The `api` service currently ends with:

```yaml
    depends_on:
      qdrant:
        condition: service_healthy

  ui:
```

Change to:

```yaml
    depends_on:
      qdrant:
        condition: service_healthy
    volumes:
      - ./eval/results:/app/eval/results

  ui:
```

(Add only this `volumes:` block to the `api` service — do not touch `qdrant`, `ui`, `ingestion`, or `eval`.)

- [ ] **Step 3: Manual verification (needs live Qdrant + LLM/judge backend)**

```bash
docker compose up -d qdrant
.venv/bin/uvicorn api.main:app --port 8000 &
sleep 3
curl -s -X POST http://localhost:8000/eval/run | python3 -m json.tool
curl -s http://localhost:8000/eval/runs | python3 -m json.tool
kill %1
```

Expected: the `POST` response is a summary object (`id`, `count: 18`, `avg_faithfulness`, `avg_correctness`); the `GET /eval/runs` response includes that same `id` in its list. If Ollama isn't reachable in this environment, fall back to a static-code check confirming the diff matches this task's steps exactly, and report which verification you actually performed.

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all 22 tests still pass (this task adds no new automated tests, but confirms nothing regressed).

- [ ] **Step 5: Commit**

```bash
git add api/eval_routes.py docker-compose.yml
git commit -m "feat: add POST /eval/run and mount eval results volume on api"
```

---

### Task 4: `chunk_document()` extraction + `POST /documents`

**Files:**
- Modify: `ingestion/ingest.py`
- Create: `api/documents_routes.py`
- Modify: `api/main.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `config.get_vectorstore()` (existing).
- Produces: `ingestion.ingest.chunk_document(text: str, source: str, path: str, url: str) -> list[dict]` — used by both the existing `chunk_pages` (refactored to call it per page) and the new upload endpoint. `api.documents_routes.router` — mounted in `main.py`.

**Context:** This is the "add a document" feature: upload replaces nothing — it upserts new chunks into the same collection `get_vectorstore()` already points at.

- [ ] **Step 1: Extract `chunk_document()` in `ingestion/ingest.py`**

Replace the existing `chunk_pages` function with:

```python
def chunk_document(text: str, source: str, path: str, url: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return [
        {"text": chunk, "source": source, "path": path, "url": url}
        for chunk in splitter.split_text(text)
    ]


def chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        chunks.extend(chunk_document(page["text"], page["source"], page["path"], page["url"]))
    return chunks
```

(`chunk_pages`'s external behavior is unchanged — same input, same output — it now delegates per-page work to `chunk_document`.)

- [ ] **Step 2: Write the test**

Create `tests/test_ingest.py`:

```python
# tests/test_ingest.py
from ingestion.ingest import chunk_document


def test_chunk_document_splits_long_text_and_attaches_metadata():
    text = "word " * 500
    chunks = chunk_document(text, source="upload", path="notes.md", url="")
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["source"] == "upload"
        assert chunk["path"] == "notes.md"
        assert chunk["url"] == ""
        assert chunk["text"]


def test_chunk_document_short_text_produces_one_chunk():
    chunks = chunk_document("Short text.", source="upload", path="short.md", url="")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text."
```

- [ ] **Step 3: Run the new test**

Run: `.venv/bin/pytest tests/test_ingest.py -v`
Expected: both tests PASS.

- [ ] **Step 4: Create `api/documents_routes.py`**

```python
# api/documents_routes.py
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from config import get_vectorstore
from ingestion.ingest import chunk_document

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_SUFFIXES = {".md", ".txt"}
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_added: int


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or '(none)'}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 5 MB)")

    text = content.decode("utf-8", errors="ignore")
    chunks = chunk_document(text, source="upload", path=file.filename, url="")
    if not chunks:
        raise HTTPException(status_code=400, detail="No content extracted from file")

    try:
        get_vectorstore().add_texts(
            texts=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "path": c["path"], "url": c["url"]} for c in chunks],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return DocumentUploadResponse(filename=file.filename, chunks_added=len(chunks))
```

(Verified empirically while writing this plan: `@router.post("")` combined with `prefix="/documents"` registers exactly `POST /documents`, matching both `/documents` and `/documents/` — no separate `/documents/` route needed.)

- [ ] **Step 5: Mount the router in `api/main.py`**

Change:

```python
from api import eval_routes
from api.dependencies import get_graph
```

to:

```python
from api import documents_routes, eval_routes
from api.dependencies import get_graph
```

Change:

```python
app = FastAPI(title="rag-lab")
app.include_router(eval_routes.router)
```

to:

```python
app = FastAPI(title="rag-lab")
app.include_router(eval_routes.router)
app.include_router(documents_routes.router)
```

- [ ] **Step 6: Manual verification (needs live Qdrant + embeddings backend)**

```bash
docker compose up -d qdrant
.venv/bin/uvicorn api.main:app --port 8000 &
sleep 3
printf '# Test Doc\n\nThis is a test document about widgets.\n' > /tmp/test_doc.md
curl -s -X POST http://localhost:8000/documents -F "file=@/tmp/test_doc.md" | python3 -m json.tool
kill %1
```

Expected: response is `{"filename": "test_doc.md", "chunks_added": 1}` (a short file produces one chunk). If Ollama isn't reachable in this environment, fall back to a static-code check confirming the diff matches this task's steps exactly, and report which verification you actually performed.

- [ ] **Step 7: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all tests pass (22 pre-existing + 2 new = 24), pristine output.

- [ ] **Step 8: Commit**

```bash
git add ingestion/ingest.py api/documents_routes.py api/main.py tests/test_ingest.py
git commit -m "feat: add document upload endpoint with additive upsert"
```

---

## Self-Review

- **Spec coverage:** all endpoints from the design (`/config`, `/eval/runs`, `/eval/runs/{run_id}`, `POST /eval/run`, `POST /documents`) are covered, plus the `summarize()`/`chunk_document()`/`get_graph()` extractions the design called for.
- **Placeholders:** none — every step has literal code, including exact verification commands and expected outputs.
- **Type consistency:** `chunk_document(text: str, source: str, path: str, url: str) -> list[dict]` matches its use in both `chunk_pages` (Task 4) and `upload_document` (Task 4, same task — no cross-task drift risk). `summarize(results: list[dict]) -> dict`'s exact key names (`count`, `avg_faithfulness`, `faithfulness_scored`, `avg_correctness`, `correctness_scored`) are used identically in `print_summary` (Task 1) and `_summary_response` (Task 2) — checked both match.
- **Circular import check:** `api/eval_routes.py` (Tasks 2-3) imports `api.dependencies.get_graph`, never `api.main` — confirmed no cycle. `api/documents_routes.py` (Task 4) imports only `config` and `ingestion.ingest`, neither of which imports anything from `api` — confirmed no cycle.
- **Scope:** Task 1 has no dependents within itself. Task 2 depends on Task 1's `summarize()`. Task 3 depends on Task 1's `get_graph()` and Task 2's `eval_routes.py`/`_summary_response`. Task 4 depends on nothing from Tasks 2-3 (only Task 1's file, `api/main.py`, which it edits further) — sequenced 1→2→3→4 for one coherent worktree history, but Task 4 could technically run right after Task 1 if needed.
- **Regression check:** `/health` and `/query` behavior is preserved byte-for-byte in intent across the `api/main.py` rewrite in Task 1 (only their surrounding imports/file structure changed, not their bodies). No task touches `graph/build.py`, `eval/judge.py`, or `ui/app.py`.
