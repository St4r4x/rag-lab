# Dashboard UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `ui/app.py`'s single chat interface into a four-tab dashboard (Chat, Évaluation, Documents, Paramètres) consuming Sub-project A's API endpoints, without changing the chat tab's existing behavior.

**Architecture:** One module per tab (`ui/chat_tab.py`, `ui/eval_tab.py`, `ui/documents_tab.py`, `ui/config_tab.py`), each exposing a `build_*_tab()` function; `ui/app.py` assembles them inside a `gr.Blocks`/`gr.Tabs` layout. Data that needs to be present when the page loads (the eval run list, the config table) is fetched via `demo.load(fn=..., outputs=...)` — a Gradio mechanism that fires when the page loads in the browser, not at Python-process start — verified empirically against the installed Gradio 6.22.0 to avoid a startup race with the `api` container.

**Tech Stack:** Gradio (already a dependency), `httpx` (already a dependency, already used by the existing chat tab). No new dependencies, no docker-compose/pyproject.toml changes.

## Global Constraints

- Type hints on all function signatures and return types (per `~/.claude/rules/python.md`)
- Prefer f-strings over `.format()`/`%`
- Use `pathlib.Path` instead of `os.path`
- Import order: stdlib → third-party → local; within each group, bare `import x` statements before `from x import y` statements, each alphabetical among themselves (matches this project's existing style in `config.py` and the current `ui/app.py` — verify against those two files if unsure, don't guess)
- Comments in English (per project `CLAUDE.md`)
- Commits: English, imperative mood, conventional-commits prefix (`feat:`/`refactor:`), max 72-char subject, no trailing period (per `~/.claude/rules/git.md`)
- Run tests with `.venv/bin/pytest tests/ -v`
- **Lesson from every prior plan on this project:** when editing an existing list/file, add without deleting or reordering existing content unless explicitly refactoring with behavior preserved. Re-read your diff line by line before committing.
- No live Qdrant/Ollama/API needed to verify any task in this plan — each task's verification is a safe, no-network Python import that builds the Gradio component tree and inspects it. This works because `demo.load(...)` registers a *browser-triggered* callback; it does not execute anything at import time.
- Pure formatting functions (`format_run_label`, `build_dropdown_choices`, `detail_to_rows`, `format_config_rows`) get automated tests. Thin HTTP-calling functions (`ask`, `fetch_runs`, `fetch_run_detail`, `trigger_run`, `upload_document`, `fetch_config`) do not — this matches the existing, already-merged `ask()` function's precedent (never had a test).

---

### Task 1: Extract `ui/chat_tab.py`, rebuild `ui/app.py` as a `Blocks`/`Tabs` skeleton

**Files:**
- Create: `ui/chat_tab.py`
- Modify: `ui/app.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ui.chat_tab.build_chat_tab() -> None` (registers a `gr.ChatInterface` in whatever `gr.Blocks`/`gr.Tab` context is currently open) — Tasks 2-4 don't depend on this, but this task establishes the pattern (`build_*_tab()`) they all follow.

**Context:** This is a behavior-preserving extraction: `ask()` moves verbatim into its own module, and `ui/app.py` becomes a `gr.Blocks` with one `gr.Tab("Chat")` for now — Tasks 2-4 add the other three tabs. The chat interface's `title=` argument moves from the `gr.ChatInterface` itself to `gr.Blocks(title=...)`, since a redundant sub-title above an already-labeled "Chat" tab would be visual clutter (this is the one intentional behavior difference — everything else about the chat flow is unchanged).

- [ ] **Step 1: Create `ui/chat_tab.py`**

```python
# ui/chat_tab.py
import os

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def ask(message: str, _history) -> str:
    response = httpx.post(f"{API_URL}/query", json={"question": message}, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not data["sources"]:
        return data["answer"]
    sources = "\n".join(f"- {s}" for s in data["sources"])
    return f"{data['answer']}\n\nSources:\n{sources}"


def build_chat_tab() -> None:
    gr.ChatInterface(fn=ask)
```

- [ ] **Step 2: Rewrite `ui/app.py`**

Replace the entire file with:

```python
# ui/app.py
import gradio as gr

from ui.chat_tab import build_chat_tab

with gr.Blocks(title="rag-lab — LangChain/LangGraph doc assistant") as demo:
    with gr.Tab("Chat"):
        build_chat_tab()

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 3: Verify**

```bash
.venv/bin/python -c "
import ui.app as app
print('title:', app.demo.title)
print('blocks:', len(app.demo.blocks))
"
```

Expected: prints the title (`rag-lab — LangChain/LangGraph doc assistant`) and a block count, with no exceptions. This confirms the module imports cleanly and the `gr.Blocks` tree builds successfully — no live Qdrant/API/Ollama needed, since nothing in this task calls `httpx` at import time (`ask` is only *defined*, not called).

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all 24 pre-existing tests still pass (this task adds no new automated tests).

- [ ] **Step 5: Commit**

```bash
git add ui/chat_tab.py ui/app.py
git commit -m "refactor: extract chat tab and rebuild ui/app.py as tabbed Blocks"
```

---

### Task 2: `ui/eval_tab.py` — evaluation dashboard tab

**Files:**
- Create: `ui/eval_tab.py`
- Modify: `ui/app.py`
- Test: `tests/test_eval_tab.py` (new)

**Interfaces:**
- Consumes: `GET /eval/runs`, `GET /eval/runs/{run_id}`, `POST /eval/run` (Sub-project A, already merged).
- Produces: `ui.eval_tab.build_eval_tab() -> tuple[gr.Dropdown, Callable[[], gr.Dropdown]]` — the dropdown component and a zero-argument loader function, both consumed by `ui/app.py`'s `demo.load(...)` wiring.

**Context:** Lists past evaluation runs in a dropdown, shows the selected run's 18-question detail in a table, and offers a button to trigger a fresh run (which can take from tens of seconds to a few minutes with a local model — Gradio's built-in button-loading state is the only progress feedback, no custom progress bar).

- [ ] **Step 1: Create `ui/eval_tab.py`**

```python
# ui/eval_tab.py
import os

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def fetch_runs() -> list[dict]:
    response = httpx.get(f"{API_URL}/eval/runs", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_run_detail(run_id: str) -> list[dict]:
    response = httpx.get(f"{API_URL}/eval/runs/{run_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def trigger_run() -> list[dict]:
    response = httpx.post(f"{API_URL}/eval/run", timeout=600)
    response.raise_for_status()
    return fetch_runs()


def format_run_label(run: dict) -> str:
    faithfulness = f"{run['avg_faithfulness']:.2f}" if run["avg_faithfulness"] is not None else "n/a"
    correctness = f"{run['avg_correctness']:.2f}" if run["avg_correctness"] is not None else "n/a"
    return f"{run['id']} — {run['count']} questions, faithfulness {faithfulness}, correctness {correctness}"


def build_dropdown_choices(runs: list[dict]) -> list[tuple[str, str]]:
    return [(format_run_label(run), run["id"]) for run in runs]


def detail_to_rows(detail: list[dict]) -> list[list]:
    return [
        [item["id"], item["category"], item["question"], item["faithfulness"], item["correctness"]]
        for item in detail
    ]


def build_eval_tab() -> tuple[gr.Dropdown, callable]:
    dropdown = gr.Dropdown(choices=[], label="Run")
    table = gr.Dataframe(
        headers=["id", "category", "question", "faithfulness", "correctness"],
        label="Détail",
    )
    refresh_button = gr.Button("Rafraîchir la liste")
    run_button = gr.Button("Lancer une évaluation")

    def on_select(run_id):
        if not run_id:
            return []
        return detail_to_rows(fetch_run_detail(run_id))

    def load_runs():
        runs = fetch_runs()
        return gr.Dropdown(choices=build_dropdown_choices(runs), value=runs[0]["id"] if runs else None)

    def on_run():
        runs = trigger_run()
        return gr.Dropdown(choices=build_dropdown_choices(runs), value=runs[0]["id"] if runs else None)

    dropdown.change(fn=on_select, inputs=dropdown, outputs=table)
    refresh_button.click(fn=load_runs, outputs=dropdown)
    run_button.click(fn=on_run, outputs=dropdown)

    return dropdown, load_runs
```

- [ ] **Step 2: Update `ui/app.py`**

Replace the entire file with:

```python
# ui/app.py
import gradio as gr

from ui.chat_tab import build_chat_tab
from ui.eval_tab import build_eval_tab

with gr.Blocks(title="rag-lab — LangChain/LangGraph doc assistant") as demo:
    with gr.Tab("Chat"):
        build_chat_tab()
    with gr.Tab("Évaluation"):
        eval_dropdown, load_runs = build_eval_tab()

    demo.load(fn=load_runs, outputs=eval_dropdown)

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 3: Write the tests**

Create `tests/test_eval_tab.py`:

```python
# tests/test_eval_tab.py
from ui.eval_tab import build_dropdown_choices, detail_to_rows, format_run_label


def test_format_run_label_with_scores():
    run = {"id": "20260101T000000Z", "count": 18, "avg_faithfulness": 3.9, "avg_correctness": 4.0}
    label = format_run_label(run)
    assert "20260101T000000Z" in label
    assert "18" in label
    assert "3.90" in label
    assert "4.00" in label


def test_format_run_label_handles_unscored():
    run = {"id": "r1", "count": 2, "avg_faithfulness": None, "avg_correctness": None}
    label = format_run_label(run)
    assert "n/a" in label


def test_build_dropdown_choices():
    runs = [{"id": "r1", "count": 1, "avg_faithfulness": 4.0, "avg_correctness": 5.0}]
    choices = build_dropdown_choices(runs)
    assert choices == [(format_run_label(runs[0]), "r1")]


def test_detail_to_rows():
    detail = [{"id": "q01", "category": "identifier", "question": "What?", "faithfulness": 4, "correctness": 5}]
    rows = detail_to_rows(detail)
    assert rows == [["q01", "identifier", "What?", 4, 5]]
```

- [ ] **Step 4: Verify**

```bash
.venv/bin/pytest tests/test_eval_tab.py -v
.venv/bin/python -c "
import ui.app as app
print('title:', app.demo.title)
"
```

Expected: 4 new tests pass; the import check prints the title with no exceptions (confirms `demo.load(...)` registration didn't raise — it only *registers* `load_runs`, it doesn't call it).

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all tests pass (24 pre-existing + 4 new = 28), pristine output.

- [ ] **Step 6: Commit**

```bash
git add ui/eval_tab.py ui/app.py tests/test_eval_tab.py
git commit -m "feat: add evaluation dashboard tab"
```

---

### Task 3: `ui/documents_tab.py` — document upload tab

**Files:**
- Create: `ui/documents_tab.py`
- Modify: `ui/app.py`

**Interfaces:**
- Consumes: `POST /documents` (Sub-project A, already merged).
- Produces: `ui.documents_tab.build_documents_tab() -> None` — nothing consumed elsewhere (no `demo.load` needed; there's no data to prefill).

**Context:** A file upload widget restricted to `.md`/`.txt` (matching the API's own restriction), an upload button, and a status textbox showing the result.

- [ ] **Step 1: Create `ui/documents_tab.py`**

```python
# ui/documents_tab.py
import os
from pathlib import Path

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def upload_document(file_path: str) -> str:
    path = Path(file_path)
    with path.open("rb") as f:
        response = httpx.post(f"{API_URL}/documents", files={"file": (path.name, f)}, timeout=60)
    response.raise_for_status()
    data = response.json()
    return f"{data['filename']} : {data['chunks_added']} chunk(s) ajouté(s)."


def build_documents_tab() -> None:
    file_input = gr.File(label="Fichier (.md ou .txt)", file_types=[".md", ".txt"])
    upload_button = gr.Button("Uploader")
    status = gr.Textbox(label="Résultat", interactive=False)

    upload_button.click(fn=upload_document, inputs=file_input, outputs=status)
```

(`gr.File`'s default `type="filepath"` — verified against the installed Gradio version — passes the uploaded file's server-side temp path as a plain string to `upload_document`, which is why `upload_document` takes `file_path: str`.)

- [ ] **Step 2: Update `ui/app.py`**

Replace the entire file with:

```python
# ui/app.py
import gradio as gr

from ui.chat_tab import build_chat_tab
from ui.documents_tab import build_documents_tab
from ui.eval_tab import build_eval_tab

with gr.Blocks(title="rag-lab — LangChain/LangGraph doc assistant") as demo:
    with gr.Tab("Chat"):
        build_chat_tab()
    with gr.Tab("Évaluation"):
        eval_dropdown, load_runs = build_eval_tab()
    with gr.Tab("Documents"):
        build_documents_tab()

    demo.load(fn=load_runs, outputs=eval_dropdown)

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 3: Verify**

```bash
.venv/bin/python -c "
import ui.app as app
print('title:', app.demo.title)
"
```

Expected: prints the title, no exceptions.

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all 28 tests still pass (this task adds no new automated tests — no pure logic to extract here, matches `ask()`'s precedent).

- [ ] **Step 5: Commit**

```bash
git add ui/documents_tab.py ui/app.py
git commit -m "feat: add document upload tab"
```

---

### Task 4: `ui/config_tab.py` — read-only parameters tab

**Files:**
- Create: `ui/config_tab.py`
- Modify: `ui/app.py`
- Test: `tests/test_config_tab.py` (new)

**Interfaces:**
- Consumes: `GET /config` (Sub-project A, already merged).
- Produces: `ui.config_tab.build_config_tab() -> tuple[gr.Dataframe, callable]` — consumed by `ui/app.py`'s `demo.load(...)` wiring, same pattern as Task 2's eval tab.

**Context:** The last tab — a read-only table of the API's `/config` response, with French display labels for each field and a refresh button.

- [ ] **Step 1: Create `ui/config_tab.py`**

```python
# ui/config_tab.py
import os

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

CONFIG_LABELS = {
    "llm_model": "Modèle LLM",
    "embedding_model": "Modèle d'embeddings",
    "sparse_embedding_model": "Modèle sparse",
    "reranker_model": "Modèle de reranking",
    "judge_model": "Modèle juge (éval)",
    "qdrant_url": "URL Qdrant",
    "qdrant_collection": "Collection Qdrant",
}


def fetch_config() -> dict:
    response = httpx.get(f"{API_URL}/config", timeout=30)
    response.raise_for_status()
    return response.json()


def format_config_rows(config: dict) -> list[list[str]]:
    return [[CONFIG_LABELS.get(key, key), value] for key, value in config.items()]


def build_config_tab() -> tuple[gr.Dataframe, callable]:
    table = gr.Dataframe(headers=["Paramètre", "Valeur"], label="Configuration actuelle")
    refresh_button = gr.Button("Rafraîchir")

    def load_config():
        return format_config_rows(fetch_config())

    refresh_button.click(fn=load_config, outputs=table)
    return table, load_config
```

- [ ] **Step 2: Update `ui/app.py`**

Replace the entire file with:

```python
# ui/app.py
import gradio as gr

from ui.chat_tab import build_chat_tab
from ui.config_tab import build_config_tab
from ui.documents_tab import build_documents_tab
from ui.eval_tab import build_eval_tab

with gr.Blocks(title="rag-lab — LangChain/LangGraph doc assistant") as demo:
    with gr.Tab("Chat"):
        build_chat_tab()
    with gr.Tab("Évaluation"):
        eval_dropdown, load_runs = build_eval_tab()
    with gr.Tab("Documents"):
        build_documents_tab()
    with gr.Tab("Paramètres"):
        config_table, load_config = build_config_tab()

    demo.load(fn=load_runs, outputs=eval_dropdown)
    demo.load(fn=load_config, outputs=config_table)

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 3: Write the tests**

Create `tests/test_config_tab.py`:

```python
# tests/test_config_tab.py
from ui.config_tab import CONFIG_LABELS, format_config_rows


def test_format_config_rows_uses_friendly_labels():
    config = {"llm_model": "ollama:llama3.2:3b", "qdrant_url": "http://localhost:6333"}
    rows = format_config_rows(config)
    assert rows == [
        [CONFIG_LABELS["llm_model"], "ollama:llama3.2:3b"],
        [CONFIG_LABELS["qdrant_url"], "http://localhost:6333"],
    ]


def test_format_config_rows_falls_back_to_key_for_unknown_field():
    rows = format_config_rows({"new_field": "value"})
    assert rows == [["new_field", "value"]]
```

- [ ] **Step 4: Verify**

```bash
.venv/bin/pytest tests/test_config_tab.py -v
.venv/bin/python -c "
import ui.app as app
print('title:', app.demo.title)
"
```

Expected: 2 new tests pass; the import check prints the title with no exceptions.

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all tests pass (28 pre-existing + 2 new = 30), pristine output.

- [ ] **Step 6: Commit**

```bash
git add ui/config_tab.py ui/app.py tests/test_config_tab.py
git commit -m "feat: add read-only parameters tab"
```

---

## Self-Review

- **Spec coverage:** all four tabs from the design (Chat extraction, Évaluation, Documents, Paramètres) are covered, plus the `demo.load` startup-race mitigation applied consistently to both tabs that need prefilled data.
- **Placeholders:** none — every step has literal code, including exact expected verification output.
- **Type consistency:** `build_eval_tab()`/`build_config_tab()` both return `tuple[Component, callable]` and are consumed identically in `ui/app.py` (`x, load_x = build_x_tab()` then `demo.load(fn=load_x, outputs=x)`) — checked both follow the same shape. `build_chat_tab()`/`build_documents_tab()` both return `None` and are called without capturing a value — consistent with having no `demo.load` need.
- **Scope:** each task rewrites `ui/app.py` in full (small file, always shown complete) rather than diffing against the previous task's version — avoids any ambiguity about exact current state between tasks. Task 1 has no dependents; Tasks 2-4 each only depend on Task 1's `ui/chat_tab.py` existing (for the import line) and the overall `Blocks`/`Tab` pattern it established, not on each other's specific code.
- **Regression check:** no task touches `api/*.py`, `graph/build.py`, `ingestion/ingest.py`, `eval/*.py`, or any existing test file — this plan only adds new `ui/*.py` modules and rewrites `ui/app.py`.
