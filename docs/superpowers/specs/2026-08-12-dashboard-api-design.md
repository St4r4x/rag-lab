# Dashboard API (Sub-project A) Design

Date: 2026-08-12
Status: approved, ready for implementation planning

## Why this document

The user asked for a web dashboard on top of `rag-lab`: eval results, adding documents, viewing the LLM-judge setup, and current parameters. This is bigger than any single prior tier, so it's split into two sub-projects: **A) API endpoints** (this document) and **B) Gradio UI tabs** consuming them (separate spec/plan, after A lands). The existing `ui/app.py` chat tab already talks to `api/main.py` over HTTP rather than importing `config`/`graph` directly — Sub-project B will follow that same pattern for the new tabs, so all the real logic belongs in the API layer, not the UI.

## Scope decisions (from user Q&A during brainstorming)

- **UI framework:** extend the existing Gradio app with tabs, not a separate service/framework.
- **Adding documents:** upload file(s) that get chunked, embedded (dense+sparse), and **upserted** into the existing Qdrant collection — nothing gets deleted (unlike `ingestion/ingest.py`'s full `force_recreate` rebuild).
- **Parameters:** read-only display of the effective current config. No hot-editing — changing a model means editing `.env` and restarting, avoiding any risk of `get_vectorstore()`/`get_reranker()`'s `lru_cache`d instances going stale mid-request.
- **Eval dashboard:** both viewing past runs and triggering a new run from the UI (Sub-project B) — this document's `POST /eval/run` is what makes the "trigger" half possible.

## File structure

`api/main.py` has grown from 2 endpoints (`/health`, `/query`) to a much larger surface (`/config` plus 4 eval/document endpoints). Splitting into routers now, while it's still small, is cheaper than after Sub-project B adds more UI-driven traffic:

- **`api/main.py`** — unchanged `/health`/`/query`, plus new `GET /config` (small enough to stay here rather than its own router file).
- **`api/eval_routes.py`** (new) — an `APIRouter` with `GET /eval/runs`, `GET /eval/runs/{run_id}`, `POST /eval/run`. Mounted in `main.py` via `app.include_router(eval_routes.router)`.
- **`api/documents_routes.py`** (new) — an `APIRouter` with `POST /documents`. Mounted the same way.
- **`eval/run_eval.py`** — gains a `summarize(results: list[dict]) -> dict` pure function, extracted from the existing `print_summary`, so both the CLI script and the new API endpoints compute averages the same way.
- **`ingestion/ingest.py`** — gains `chunk_document(text: str, source: str, path: str, url: str) -> list[dict]`, extracted from the existing `chunk_pages` (which becomes a thin loop calling it per page) — the same per-document chunking logic the new `POST /documents` endpoint needs.
- **`docker-compose.yml`** — the `api` service gets the same `./eval/results:/app/eval/results` bind mount the `eval` service already has, so both can see/write the same reports.

## Endpoint details

### `GET /config`

Reads env vars directly (same defaults `config.py` already uses) — does **not** construct any LLM/embedding client just to report a model name. Never reads or returns anything resembling an API key. Response: `llm_model`, `embedding_model`, `sparse_embedding_model`, `reranker_model`, `judge_model` (resolves `EVAL_JUDGE_MODEL` falling back to `llm_model`, matching `config.get_judge_llm()`'s actual fallback), `qdrant_url`, `qdrant_collection`.

### `GET /eval/runs` and `GET /eval/runs/{run_id}`

Lists/reads `eval/results/*.json`. **Security note:** `run_id` comes from the URL path — validated against `^\d{8}T\d{6}Z$` (the exact timestamp format `write_report` already generates) *before* it touches the filesystem, so path traversal (`../../etc/passwd`) is rejected by construction rather than by an after-the-fact containment check. Listing returns a summary per run (via `summarize()`); the detail endpoint returns the full per-question list.

### `POST /eval/run`

Runs a fresh evaluation synchronously and returns its summary. Reuses `api/main.py`'s already-cached `get_graph()` (the same graph the chat endpoint uses — no second graph gets built) plus `eval/run_eval.py`'s existing `load_dataset`/`evaluate_one`/`write_report`. No new orchestration logic, just wiring.

### `POST /documents`

Multipart file upload, `.md`/`.txt` only (no PDF — no expressed need, would add a parsing dependency), capped at 5 MB. Chunks via the new `chunk_document()` helper with `source="upload"`, `path=<filename>`, `url=""` (uploaded files have no public URL). Upserts via `get_vectorstore().add_texts(texts=..., metadatas=...)` — the vectorstore instance already knows its own dense+sparse embedding models (configured in `config.get_vectorstore()`), so no embedding objects need to be passed explicitly.

## Testing

New `tests/test_run_eval.py`: `summarize()` on known inputs (all scored, partially scored, none scored).
New `tests/test_ingest.py`: `chunk_document()` on known input text, checking chunk count and metadata shape.
New `tests/test_api.py`: `GET /config` (via `fastapi.testclient.TestClient`) returns expected fields and — a deliberate negative check — contains no substring that looks like a secret (`key`/`token`); `GET /eval/runs`/`GET /eval/runs/{run_id}` against a temp results directory (`monkeypatch`); `run_id` validation rejects malformed/traversal-style ids with 400 before any file access, and returns 404 for a well-formed-but-missing id.

`POST /eval/run` and `POST /documents` have no automated tests — both need a live Qdrant + LLM/embeddings backend, same precedent as `ingestion/ingest.py` and `eval/run_eval.py`. Verified manually.

Using `fastapi.testclient.TestClient` surfaces a known, non-actionable `StarletteDeprecationWarning` ("install httpx2 instead") unrelated to this project's code — `pyproject.toml` gets a `[tool.pytest.ini_options]` `filterwarnings` entry to ignore exactly that warning class, keeping test output pristine without swallowing anything else.

## Explicitly out of scope for this pass

- The Gradio UI tabs themselves (Sub-project B, next).
- PDF or other non-text document upload formats.
- Deleting/replacing individual uploaded documents (only additive upsert for now).
- Hot-reloading config without a restart.
