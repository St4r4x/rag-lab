# Tier 2 Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, dependency-light evaluation harness — a golden Q&A dataset, an LLM-judge scoring module, and a runner script — so future retrieval/generation changes can be measured against a saved baseline instead of judged by feel.

**Architecture:** A new `eval/` package (sibling to `ingestion/`, `graph/`, `api/`, `ui/`). `eval/golden_dataset.json` holds 18 hand-written Q&A pairs. `eval/judge.py` holds two pure, testable LLM-judge functions (faithfulness, correctness), each one LLM call parsed for a single digit 1-5. `eval/run_eval.py` orchestrates: build the real `build_graph_v2` graph, run every question through it, score with the judge, print a summary, and write a timestamped JSON report. `config.py` gains `get_judge_llm()`. `docker-compose.yml` gains an `eval` service matching the existing `ingestion` service's shape.

**Tech Stack:** Python 3.11+, LangChain/LangGraph (already a dependency), `pytest`. No new dependencies.

## Global Constraints

- Type hints on all function signatures and return types (per `~/.claude/rules/python.md`)
- Prefer f-strings over `.format()`/`%`
- Use `pathlib.Path` instead of `os.path`
- Import order: stdlib → third-party → local, alphabetical within each group
- Never use bare `except:` — always specify exception type
- Comments in English (per project `CLAUDE.md`)
- Commits: English, imperative mood, conventional-commits prefix (`feat:`/`docs:`), max 72-char subject, no trailing period (per `~/.claude/rules/git.md`)
- Run tests with `.venv/bin/pytest tests/ -v` (matches `.github/workflows/ci.yml`)
- **Lesson from the Tier 1 plan on this project:** when editing an existing list (imports, dependencies, docker-compose services), add new entries without deleting or reordering existing ones unless explicitly told to. Re-read your diff line by line before committing to confirm this.
- No dedicated automated test for changes that need a live Qdrant/Ollama backend or that mirror an existing untested factory pattern in the same file — this plan says explicitly, per task, where that applies

---

### Task 1: Golden dataset + judge LLM config factory

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/golden_dataset.json`
- Modify: `config.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing new.
- Produces: `config.get_judge_llm()` — Task 3 (`run_eval.py`) calls this for scoring, separately from `get_llm()` (used for the graph itself). `eval/golden_dataset.json` — Task 3 loads and parses this as a JSON array of `{"id": str, "category": str, "question": str, "reference_answer": str}` objects.

**Context:** `config.py` currently has `get_llm()`, `get_embeddings()`, `get_sparse_embeddings()`, `get_vectorstore()`, `get_reranker()` — each a thin factory, none with a dedicated test. This task follows that exact pattern for `get_judge_llm()`.

- [ ] **Step 1: Create the `eval` package**

Create `eval/__init__.py` (empty file, matching `ingestion/__init__.py`, `graph/__init__.py`, etc.).

- [ ] **Step 2: Create the golden dataset**

Create `eval/golden_dataset.json` with exactly this content:

```json
[
  {"id": "q01", "category": "identifier", "question": "What class do you use in LangGraph to define a stateful, cyclic computation graph?", "reference_answer": "StateGraph — you instantiate it with a state schema (e.g. a TypedDict) and add nodes and edges to it."},
  {"id": "q02", "category": "identifier", "question": "Which method turns a LangGraph StateGraph into an executable object?", "reference_answer": "compile() — calling graph.compile() on a StateGraph returns a runnable compiled graph that supports invoke(), stream(), etc."},
  {"id": "q03", "category": "identifier", "question": "What are the two special node identifiers used to mark the entry and exit points of a LangGraph graph?", "reference_answer": "START and END, imported from langgraph.graph."},
  {"id": "q04", "category": "identifier", "question": "Which method do you call on a StateGraph to add a node that routes to different next nodes based on a function's return value?", "reference_answer": "add_conditional_edges — it takes a source node, a routing function, and a mapping from the function's return values to destination node names."},
  {"id": "q05", "category": "identifier", "question": "What LangGraph component lets a graph persist and resume its state across runs, for memory or human-in-the-loop?", "reference_answer": "A checkpointer, e.g. MemorySaver for in-memory persistence, passed to .compile(checkpointer=...)."},
  {"id": "q06", "category": "conceptual", "question": "In LangChain, what is the base abstraction that represents anything invokable — LLMs, chains, tools, retrievers — with a common invoke/stream/batch interface?", "reference_answer": "A Runnable (the Runnable protocol / LangChain Expression Language, LCEL)."},
  {"id": "q07", "category": "conceptual", "question": "What operator does LangChain's expression language (LCEL) overload to chain Runnables together?", "reference_answer": "The pipe operator |, e.g. prompt | llm | output_parser."},
  {"id": "q08", "category": "identifier", "question": "Which LangChain helper wraps an arbitrary Python function so it can be used as a Runnable in a chain?", "reference_answer": "RunnableLambda."},
  {"id": "q09", "category": "identifier", "question": "Which LangChain Runnable runs multiple Runnables concurrently on the same input and returns a dict of their outputs?", "reference_answer": "RunnableParallel."},
  {"id": "q10", "category": "conceptual", "question": "What's the difference between invoke() and stream() on a LangChain Runnable?", "reference_answer": "invoke() runs the Runnable and returns the complete output once finished; stream() returns an iterator that yields output chunks incrementally as they're produced."},
  {"id": "q11", "category": "identifier", "question": "What function does LangChain provide to construct a chat model instance from a provider:model string, like \"openai:gpt-4o-mini\", without importing a provider-specific class directly?", "reference_answer": "init_chat_model."},
  {"id": "q12", "category": "conceptual", "question": "In a LangGraph state schema, what do you use to tell the graph how to combine a field's old and new values across node updates, instead of overwriting it?", "reference_answer": "An Annotated type with a reducer function, e.g. Annotated[list, operator.add], or add_messages for message lists."},
  {"id": "q13", "category": "conceptual", "question": "What LangGraph feature lets a graph pause execution mid-run and wait for external input before continuing, for human-in-the-loop workflows?", "reference_answer": "Interrupts — the interrupt() function inside a node, combined with a checkpointer, pauses the graph so it can be resumed later with a Command(resume=...)."},
  {"id": "q14", "category": "identifier", "question": "How do you add a node to a LangGraph StateGraph?", "reference_answer": "graph.add_node(\"name\", function) — the function takes the current state and returns a partial state update dict."},
  {"id": "q15", "category": "identifier", "question": "How do you connect two nodes with a plain, unconditional edge in LangGraph?", "reference_answer": "graph.add_edge(\"source_node\", \"destination_node\")."},
  {"id": "q16", "category": "conceptual", "question": "What is the recommended way to give a LangChain chat model access to external functions it can call?", "reference_answer": "Tool calling — bind tools to the model with .bind_tools([...]), and the model returns tool calls in its response for your code to execute."},
  {"id": "q17", "category": "out_of_scope", "question": "What is the capital of France?", "reference_answer": "Outside the scope of the LangChain/LangGraph documentation — the RAG system should say it doesn't have enough information rather than guessing."},
  {"id": "q18", "category": "out_of_scope", "question": "How do you configure liveness and readiness probes in a Kubernetes Deployment YAML?", "reference_answer": "Outside the scope of the LangChain/LangGraph documentation — the RAG system should say it doesn't have enough information rather than answering from general knowledge."}
]
```

- [ ] **Step 3: Add `get_judge_llm()` to `config.py`**

Add this function right after `get_llm()` (which is currently the first factory function in the file):

```python
def get_judge_llm():
    model = os.environ.get("EVAL_JUDGE_MODEL", "")
    if not model:
        return get_llm()
    return init_chat_model(model, **_ollama_kwargs(model))
```

No new imports needed — `os`, `init_chat_model`, and `_ollama_kwargs` are already imported/defined in this file.

- [ ] **Step 4: Register the `eval` package for packaging**

In `pyproject.toml`, change:

```toml
[tool.setuptools.packages.find]
include = ["ingestion*", "graph*", "api*", "ui*"]
```

to:

```toml
[tool.setuptools.packages.find]
include = ["ingestion*", "graph*", "api*", "ui*", "eval*"]
```

(Add `"eval*"` to the existing list — do not remove or reorder the other three entries.)

- [ ] **Step 5: Verify**

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -c "
import json
from pathlib import Path
from config import get_judge_llm

data = json.loads(Path('eval/golden_dataset.json').read_text(encoding='utf-8'))
print(f'{len(data)} questions loaded')
print(data[0])
print(type(get_judge_llm()).__name__)
"
```

Expected: prints `18 questions loaded`, the first question dict, and a chat model class name (e.g. `ChatOllama` — confirms `get_judge_llm()` falls back to `get_llm()`'s configured model when `EVAL_JUDGE_MODEL` is unset).

- [ ] **Step 6: Commit**

```bash
git add eval/__init__.py eval/golden_dataset.json config.py pyproject.toml
git commit -m "feat: add eval golden dataset and judge LLM config factory"
```

---

### Task 2: LLM-judge scoring functions (TDD)

**Files:**
- Create: `eval/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: nothing new (takes an `llm`-like object with `.invoke(prompt) -> response` as a parameter, matching the dependency-injection style already used by `graph/build.py`'s `make_grade_documents(llm)` etc.).
- Produces: `score_faithfulness(llm, question, context, answer) -> int | None` and `score_correctness(llm, question, reference_answer, answer) -> int | None` — Task 3's `run_eval.py` calls both, once per evaluated question.

**Context:** These are the two LLM-judge calls described in the design: faithfulness checks the answer against the *retrieved context* (catches hallucination), correctness checks the answer against the *reference answer* (catches wrong-context-but-internally-consistent answers). Both parse a single digit 1-5 out of the LLM's raw text response; `None` means the response couldn't be parsed (not a crash — `run_eval.py` will report it as unscored).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judge.py`:

```python
# tests/test_judge.py
from types import SimpleNamespace

from eval.judge import score_correctness, score_faithfulness


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, _prompt):
        return SimpleNamespace(content=self.responses.pop(0))


def test_score_faithfulness_parses_digit_from_response():
    llm = FakeLLM(["4"])
    score = score_faithfulness(llm, "q", "context text", "answer text")
    assert score == 4


def test_score_faithfulness_returns_none_on_unparseable_response():
    llm = FakeLLM(["I cannot determine this."])
    score = score_faithfulness(llm, "q", "context text", "answer text")
    assert score is None


def test_score_correctness_parses_digit_from_response():
    llm = FakeLLM(["5"])
    score = score_correctness(llm, "q", "reference answer", "generated answer")
    assert score == 5


def test_score_correctness_extracts_digit_from_verbose_response():
    llm = FakeLLM(["I would rate this a 3 out of 5."])
    score = score_correctness(llm, "q", "reference answer", "generated answer")
    assert score == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.judge'` (or `ImportError`) — the module doesn't exist yet.

- [ ] **Step 3: Implement `eval/judge.py`**

```python
# eval/judge.py
import re


def _parse_score(text: str) -> int | None:
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else None


def score_faithfulness(llm, question: str, context: str, answer: str) -> int | None:
    response = llm.invoke(
        "You are evaluating whether an answer is faithful to (i.e., supported by) the given "
        "context. Rate faithfulness from 1 (answer is unsupported or contradicts the context) "
        "to 5 (answer is fully supported by the context). Respond with only a single digit "
        "from 1 to 5.\n\n"
        f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
    )
    return _parse_score(response.content)


def score_correctness(llm, question: str, reference_answer: str, answer: str) -> int | None:
    response = llm.invoke(
        "You are evaluating whether a generated answer correctly addresses a question, "
        "compared to a reference answer. Rate correctness from 1 (wrong or irrelevant) to "
        "5 (fully correct and complete). Respond with only a single digit from 1 to 5.\n\n"
        f"Question:\n{question}\n\nReference answer:\n{reference_answer}\n\n"
        f"Generated answer:\n{answer}"
    )
    return _parse_score(response.content)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_judge.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all tests pass (the pre-existing 8 plus these 4 new ones = 12), pristine output.

- [ ] **Step 6: Commit**

```bash
git add eval/judge.py tests/test_judge.py
git commit -m "feat: add faithfulness and correctness LLM-judge scoring"
```

---

### Task 3: Eval runner, docker-compose integration, docs

**Files:**
- Create: `eval/run_eval.py`
- Modify: `docker-compose.yml`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: `config.get_llm()`, `config.get_vectorstore()`, `config.get_reranker()`, `config.get_judge_llm()` (Task 1), `graph.build.build_graph_v2` (existing), `eval.judge.score_faithfulness`/`score_correctness` (Task 2), `eval/golden_dataset.json` (Task 1).
- Produces: nothing consumed elsewhere — this is the top-level entry point.

**Context:** This is the orchestration script that ties Tasks 1 and 2 together into something runnable. It needs a live Qdrant + Ollama (or configured API) backend, same as `ingestion/ingest.py` — no automated test for the script itself, only manual verification.

- [ ] **Step 1: Create `eval/run_eval.py`**

```python
# eval/run_eval.py
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from config import get_judge_llm, get_llm, get_reranker, get_vectorstore
from eval.judge import score_correctness, score_faithfulness
from graph.build import build_graph_v2

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def evaluate_one(graph, judge_llm, item: dict) -> dict:
    result = graph.invoke(
        {"question": item["question"], "documents": [], "generation": "", "retries": 0}
    )
    answer = result["generation"]
    context = "\n\n".join(doc.page_content for doc in result["documents"])
    sources = sorted({doc.metadata.get("url", "") for doc in result["documents"]})

    faithfulness = score_faithfulness(judge_llm, item["question"], context, answer)
    correctness = score_correctness(judge_llm, item["question"], item["reference_answer"], answer)

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": answer,
        "sources": sources,
        "faithfulness": faithfulness,
        "correctness": correctness,
    }


def write_report(results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out_path


def print_summary(results: list[dict]) -> None:
    faithfulness_scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    correctness_scores = [r["correctness"] for r in results if r["correctness"] is not None]
    print()
    if faithfulness_scores:
        print(
            f"Average faithfulness: {statistics.mean(faithfulness_scores):.2f} "
            f"({len(faithfulness_scores)}/{len(results)} scored)"
        )
    if correctness_scores:
        print(
            f"Average correctness: {statistics.mean(correctness_scores):.2f} "
            f"({len(correctness_scores)}/{len(results)} scored)"
        )


def main() -> None:
    llm = get_llm()
    judge_llm = get_judge_llm()
    graph = build_graph_v2(llm, get_vectorstore(), get_reranker())

    results = [evaluate_one(graph, judge_llm, item) for item in load_dataset()]
    for r in results:
        print(f"[{r['id']}] faithfulness={r['faithfulness']} correctness={r['correctness']} — {r['question']}")

    out_path = write_report(results)
    print(f"\nWrote {out_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `.gitignore` entry**

In `.gitignore`, add a new line:

```
eval/results/
```

(Add this as a new line at the end of the existing file — do not remove or reorder the existing entries: `.venv/`, `__pycache__/`, `*.pyc`, `.env`, `docker-compose.override.yml`, `*.egg-info/`.)

- [ ] **Step 3: Add the `eval` service to `docker-compose.yml`**

Add this service after the existing `ingestion` service (same file, same shape — do not modify the `qdrant`, `api`, `ui`, or `ingestion` services):

```yaml
  eval:
    build: .
    command: python -m eval.run_eval
    profiles: ["tools"]
    env_file:
      - path: .env
        required: false
    environment:
      QDRANT_URL: http://qdrant:6333
      OLLAMA_BASE_URL: http://host.docker.internal:11434
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      qdrant:
        condition: service_healthy
```

- [ ] **Step 4: Document it in `README.md`**

In the "Structure" section's code block, add a line for `eval/` (after the `tests/` line):

```
tests/       # test du routing du graphe v2 (seule suite automatisée, décision volontaire)
eval/        # harnais d'évaluation : 18 Q/A, LLM-judge (faithfulness/correctness), docker compose run --rm eval
```

In the "Quickstart" section, after the `docker compose run --rm ingestion` line, add:

```bash
docker compose run --rm eval               # évalue le pipeline sur 18 questions, écrit un rapport JSON dans eval/results/
```

- [ ] **Step 5: Manual verification (needs live Qdrant + embeddings/LLM backend, same constraint as ingestion)**

```bash
docker compose up -d qdrant
.venv/bin/python -m eval.run_eval
```

Expected: 18 lines of per-question output (`[qNN] faithfulness=X correctness=Y — question text`), a `Wrote eval/results/<timestamp>.json` line, and a final summary with average faithfulness/correctness. If Ollama isn't reachable in this environment, fall back to a static-code check confirming the diff matches this task's steps exactly, and report which verification you actually performed — do not claim the runtime check passed if it didn't run.

- [ ] **Step 6: Run the full test suite one more time**

Run: `.venv/bin/pytest tests/ -v`
Expected: all 12 tests still pass (this task adds no new automated tests, but confirms nothing regressed).

- [ ] **Step 7: Commit**

```bash
git add eval/run_eval.py docker-compose.yml .gitignore README.md
git commit -m "feat: add eval runner script and docker-compose integration"
```

---

## Self-Review

- **Spec coverage:** the design's golden dataset, judge functions, runner script, config factory, and docker-compose integration are all covered — Task 1 (dataset + config), Task 2 (judge), Task 3 (runner + integration + docs).
- **Placeholders:** none — every step has literal code/content, including the full 18-question dataset.
- **Type consistency:** `score_faithfulness`/`score_correctness` both take `(llm, question: str, ..., answer: str) -> int | None`, matching how `run_eval.py`'s `evaluate_one` calls them. `get_judge_llm()` has no declared return type (matches the existing untyped convention of `get_llm()`/`get_embeddings()` in the same file — not a regression, a pre-existing pattern).
- **Scope:** Task 1 has no dependents within itself; Tasks 2 and 3 depend on Task 1 only (Task 3 also depends on Task 2's `judge.py`). Sequenced 1→2→3 for one coherent worktree history.
- **Regression check:** no step modifies `graph/build.py`, `api/main.py`, `ingestion/ingest.py`, or any existing test file — this plan only adds new files and appends to `config.py`, `pyproject.toml`, `docker-compose.yml`, `.gitignore`, `README.md`.
