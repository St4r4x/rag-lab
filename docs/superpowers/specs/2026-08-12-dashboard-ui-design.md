# Dashboard UI (Sub-project B) Design

Date: 2026-08-12
Status: approved, ready for implementation planning

## Why this document

Sub-project A (the API endpoints: `GET /config`, `GET /eval/runs`, `GET /eval/runs/{run_id}`, `POST /eval/run`, `POST /documents`) is merged into `main`. This document scopes Sub-project B: the Gradio UI tabs that consume those endpoints, completing the dashboard the user asked for.

## Approach

`ui/app.py` currently builds one `gr.ChatInterface` directly. It grows to four tabs, so — matching the router split already done for `api/main.py` in Sub-project A — it splits into one module per tab, each exposing a `build_*_tab()` function, with a thin `ui/app.py` assembling them inside a `gr.Blocks`/`gr.Tabs` layout.

**Startup-race avoidance (verified empirically against the installed Gradio 6.22.0):** fetching the eval-run list or the config at Python-process/module-load time would race the `api` container's actual readiness — `docker-compose.yml`'s `ui` service has `depends_on: - api` with no health-check condition, so `api`'s container can be *started* without its FastAPI process being ready to serve yet. Using `demo.load(fn=..., outputs=...)` instead defers those calls to when the page loads in the browser, by which point the API has had time to come up, and confirmed working via a local smoke test with `gr.ChatInterface` nested in a `gr.Tab`, `gr.Dropdown` with tuple `(label, value)` choices, `gr.Dataframe`, and `demo.load(...)` all together in one `gr.Blocks`.

## File structure

- **`ui/chat_tab.py`** — `ask()` extracted verbatim from the current `ui/app.py` (unchanged behavior), plus `build_chat_tab()` wrapping it in a `gr.ChatInterface`.
- **`ui/eval_tab.py`** — `fetch_runs()`, `fetch_run_detail(run_id)`, `trigger_run()` (thin HTTP calls, untested — matches `ask()`'s existing precedent), plus pure formatting functions `format_run_label(run) -> str`, `build_dropdown_choices(runs) -> list[tuple[str, str]]`, `detail_to_rows(detail) -> list[list]` (tested). `build_eval_tab()` wires a `gr.Dropdown` (run picker), a `gr.Dataframe` (selected run's 18-row detail), a "Rafraîchir la liste" button, and a "Lancer une évaluation" button (calls `POST /eval/run`, then refreshes the dropdown) — returns `(dropdown, load_runs)` so `ui/app.py` can wire the `demo.load` trigger.
- **`ui/documents_tab.py`** — `upload_document(file_path) -> str` (multipart upload via `httpx`, untested — same class of thin I/O as `ask()`), `build_documents_tab()` wiring a `gr.File` (default `type="filepath"`, restricted to `.md`/`.txt`) and an upload button.
- **`ui/config_tab.py`** — `fetch_config() -> dict`, pure `format_config_rows(config) -> list[list[str]]` (tested, with a `CONFIG_LABELS` dict mapping API field names to French display labels, falling back to the raw key for anything unmapped), `build_config_tab()` wiring a `gr.Dataframe` and a refresh button — returns `(table, load_config)`.
- **`ui/app.py`** — thin: builds `gr.Blocks(title="rag-lab — LangChain/LangGraph doc assistant")` with four `gr.Tab`s ("Chat", "Évaluation", "Documents", "Paramètres"), then registers two `demo.load(...)` calls (one for the eval dropdown, one for the config table) after all tabs are built.

No `docker-compose.yml`, `pyproject.toml`, or `.env.example` changes needed — `ui*` is already registered for packaging (these are new modules inside the existing `ui` package, not a new package), and the `ui` service already has `RAG_API_URL` and `depends_on: api`.

## UX notes

- The page title moves from `gr.ChatInterface(title=...)` to `gr.Blocks(title=...)` — with four tabs now, a redundant sub-title above the already-labeled "Chat" tab would be visual clutter.
- "Lancer une évaluation" has no custom progress bar — the API call is synchronous and the eval harness doesn't report incremental progress. Gradio's built-in button-loading state (automatic on any `.click()` handler) is the only feedback; this can take from tens of seconds to a few minutes with a local model, matching what the API design doc already flagged as a UI concern to budget for.
- Uploaded documents get `metadata.url = ""` (per Sub-project A's design) — the existing chat tab's source-list rendering (`if not data["sources"]: ...` then joins with `- {s}`) would render a bare `-` bullet if an uploaded chunk is ever cited. Out of scope to fix here (would touch `ui/chat_tab.py`'s already-correct-behavior extraction and `api/main.py`'s `/query`, neither of which this plan should touch) — noted as a known follow-up, not blocking.

## Testing

`tests/test_eval_tab.py`: `format_run_label` (scored and unscored/`None` cases), `build_dropdown_choices`, `detail_to_rows`.
`tests/test_config_tab.py`: `format_config_rows` (known field via `CONFIG_LABELS`, unknown field falls back to the raw key).
No tests for `ui/chat_tab.py` or `ui/documents_tab.py` — both are thin HTTP-calling glue, matching the existing untested precedent of `ask()` (never had a test, even before this plan). No tests for `ui/app.py` itself (Blocks assembly, not logic).

## Explicitly out of scope for this pass

- Fixing the blank-source-bullet issue for uploaded documents (belongs to `/query`+`ui/chat_tab.py`, a different concern than adding tabs).
- A real progress bar for evaluation runs (would need the API to report incremental progress, which it doesn't).
- Any docker-compose, packaging, or config changes (none needed).
