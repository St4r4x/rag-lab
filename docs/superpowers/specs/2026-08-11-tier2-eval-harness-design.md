# Tier 2 — Evaluation Harness Design

Date: 2026-08-11
Status: approved, ready for implementation planning

## Why this document

Tier 1 (hybrid search + reranking) is merged into `main`, judged only qualitatively per its own design doc: "this pass is judged qualitatively... until Tier 2 exists." This document scopes Tier 2 (roadmap item 7 in [2026-08-11-advanced-rag-improvements-design.md](2026-08-11-advanced-rag-improvements-design.md)): a small evaluation harness so future retrieval/generation changes can be compared against a saved baseline instead of guessed at.

## Approach: custom LLM-judge script, not RAGAS

Reuses `config.get_llm()` — no new heavy dependency, fully transparent scoring logic, consistent with this project's existing pattern of simple single-purpose LLM calls (`grade_documents`, `rewrite_query` each already do one plain LLM judgment per call). RAGAS was considered and rejected: it's a non-trivial new dependency, and its metrics' correlation with human judgment is only moderate (~0.55, per the Tier 0 research doc) — not clearly better for a project whose whole point is understanding each mechanism directly.

Two independent single-call judgments per question (not one combined multi-field call): a single "respond with one digit 1-5" question is far more reliable to parse from a small local model (`llama3.2:3b`) than asking for multiple structured fields at once.

- **Faithfulness** (1-5): is the generated answer supported by the *retrieved* context? Catches hallucination.
- **Correctness** (1-5): does the generated answer match the *reference* answer for the question? Catches the case where retrieval pulled the wrong context but the model still answered "faithfully" to that wrong context — faithfulness alone wouldn't catch this.

A configurable judge model (`EVAL_JUDGE_MODEL` env var, same pattern as `LLM_MODEL`) lets the judge be a stronger model than the one being evaluated, without forcing it — defaults to reusing `get_llm()`.

## Golden dataset

18 hand-written question/reference-answer pairs about LangChain/LangGraph, stored as `eval/golden_dataset.json`. Weighted toward exact-identifier questions (11 of 18) since that's specifically what hybrid search (BM25 sparse + dense) is supposed to help with over dense-only retrieval; the rest are conceptual questions plus 2 deliberately out-of-scope questions to confirm the "insufficient information" fallback still fires correctly.

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

## Architecture

New `eval/` package, alongside `ingestion/`, `graph/`, `api/`, `ui/`:

- **`eval/golden_dataset.json`** — the 18 Q&A pairs above.
- **`eval/judge.py`** — `score_faithfulness(llm, question, context, answer) -> int | None` and `score_correctness(llm, question, reference_answer, answer) -> int | None`, each one LLM call asking for a single digit 1-5, parsed with a small `_parse_score(text) -> int | None` helper (first character matching `[1-5]`, `None` if none found — a malformed judge response is logged as an unscored item, not a crash).
- **`eval/run_eval.py`** — loads the dataset, builds `build_graph_v2(get_llm(), get_vectorstore(), get_reranker())`, runs each question through it, scores with `judge.py` using a separate `get_judge_llm()` (defaults to `get_llm()` unless `EVAL_JUDGE_MODEL` is set), prints a per-question line and a final summary (average faithfulness/correctness, with count of unscored items), and writes a timestamped JSON report to `eval/results/` (gitignored — generated artifacts, not source).
- **`config.py`**: new `get_judge_llm()`, mirroring `get_llm()`.
- **`docker-compose.yml`**: new `eval` service under `profiles: ["tools"]`, same shape as the existing `ingestion` service (`docker compose run --rm eval`).

## Testing

`eval/judge.py`'s parsing logic is pure and testable with a `FakeLLM` (same pattern as `tests/test_graph_routing.py`, duplicated locally in the new `tests/test_judge.py` rather than shared via a new `conftest.py` — a few duplicated lines beats introducing shared test infrastructure for one small file). Covers: clean digit response, verbose response with an embedded digit, and unparseable response → `None`.

The end-to-end run (`run_eval.py` against live Qdrant + Ollama) has no automated test — same precedent as `ingestion/ingest.py` — verified manually.

## Explicitly out of scope for this pass

- Retrofitting a "before Tier 1" baseline run — Tier 1 is already merged, so this harness's first real job is establishing a going-forward baseline, not retroactively proving Tier 1 helped (already noted as a limitation in the Tier 1 design doc).
- Context precision/recall metrics against a hand-labeled "correct chunks" ground truth — expensive to build and maintain for 18 questions; the report's per-question `sources` list (retrieved doc URLs) gives enough signal for manual inspection instead.
- CI integration / pass-fail thresholds — this is a qualitative/comparative tool for a lab project, not a merge gate.
