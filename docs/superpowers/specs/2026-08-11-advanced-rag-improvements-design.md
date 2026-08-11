# Advanced RAG — Research & Improvement Roadmap for rag-lab

Date: 2026-08-11
Status: research + proposal, not yet approved for implementation

## Why this document

`rag-lab` is a learning project built around a self-RAG-lite `StateGraph`
(`retrieve → grade_documents → rewrite_query → generate`, bounded to 2
retries). It works, but it uses the simplest version of every stage: dense-only
retrieval, whole-batch binary grading, fixed-size chunking, no evaluation, no
observability. This document surveys current (2025-2026) advanced RAG
techniques and proposes which ones are worth adding here, in what order, and
why — sized to a lab project running a 3B local model over ~4000 chunks of
LangChain/LangGraph docs, not a production system.

## Current pipeline (baseline, as of this commit)

| Stage | Current implementation | File |
|---|---|---|
| Chunking | Fixed-size `RecursiveCharacterTextSplitter(800, 100)`, no document context | [ingestion/ingest.py](../../../ingestion/ingest.py) |
| Indexing | `QdrantVectorStore.from_texts()` — **appends**, never upserts; re-running ingestion duplicates every point (already flagged as a ponytail TODO) | [ingestion/ingest.py:93](../../../ingestion/ingest.py) |
| Retrieval | Dense-only `similarity_search(k=4)`, single query, no reranking | [graph/build.py:8](../../../graph/build.py) |
| Grading | One LLM call over the **concatenation of all 4 docs** → binary "yes/no" → keeps all or drops all (not per-document) | [graph/build.py:45](../../../graph/build.py) |
| Correction | Query rewrite + retry loop, bounded at 2 attempts (this is a real, working CRAG-lite pattern) | [graph/build.py:67](../../../graph/build.py) |
| Generation | Single prompt, context is the raw concatenated chunks | [graph/build.py:14](../../../graph/build.py) |
| Evaluation | None — one unit test covers only the graph's routing logic, not retrieval/answer quality | [tests/test_graph_routing.py](../../../tests/test_graph_routing.py) |
| Observability | None (no tracing) | — |
| Serving | Stateless, blocking `/query`; Gradio `ChatInterface` ignores `_history` entirely (no multi-turn memory) | [api/main.py](../../../api/main.py), [ui/app.py](../../../ui/app.py) |

## Research summary (2025-2026 state of the art)

Sources are linked inline; full list at the bottom.

- **Contextual Retrieval** (Anthropic, industry-adopted): prepend a short
  (50-100 token) LLM-generated summary of "where this chunk sits in its
  document" to each chunk before embedding *and* before BM25 indexing. Reported
  ~49% reduction in retrieval failures, ~67% combined with reranking.
- **Hybrid search (dense + sparse) with Reciprocal Rank Fusion**: dense vectors
  catch paraphrase/synonyms, sparse (BM25/SPLADE) catches exact
  identifiers/API names — which matters a lot for a corpus that's full of
  function names and code identifiers like `RunnableParallel` or
  `StateGraph`. Qdrant supports this natively via its Query API. Benchmarks
  cited: +7.4% NDCG hybrid vs either alone; Recall@5 0.816 vs 0.587 dense-only
  on one dataset.
- **Cross-encoder reranking**: second-stage precision pass over the top
  20-50 hybrid candidates, ~100-300ms added latency, consistently improves
  nDCG.
- **CRAG (Corrective RAG)**: a retrieval evaluator grades *each* document
  individually and routes accordingly — this is the correct version of what
  `grade_documents` is currently trying to do (it grades the whole batch as
  one blob today).
- **Multi-query / HyDE / query decomposition**: useful for multi-hop
  questions; diminishing returns reported past 2 retrieval iterations, and
  quality depends on the rewriting LLM being decent — a real risk with
  `llama3.2:3b`.
- **RAGAS / LLM-as-judge evaluation**: faithfulness, context precision/recall,
  answer relevancy. Documented caveat: correlation with human judgment is
  moderate (~0.55), so treat scores as a regression signal, not ground truth.
- **GraphRAG / RAPTOR**: hierarchical summarization or knowledge-graph
  indexing. High value for multi-hop reasoning over large, interconnected
  corpora; "prohibitively expensive" per multiple sources for what doesn't
  need it. This corpus (flat technical docs, ~4000 chunks) doesn't need it.
- **LangSmith/Langfuse tracing**: near-zero-effort with LangChain (env vars),
  gives full visibility into what each graph node actually retrieved/decided —
  valuable specifically *because* this is a learning project.

## Proposed roadmap (ordered by effort-to-value ratio, not by "advancedness")

### Tier 0 — fix what's already broken (do first, small diffs)

1. **Idempotent ingestion.** Replace `from_texts()` (blind append) with
   deterministic point IDs (e.g. hash of `path + chunk index`) and
   upsert. Root-cause fix for the duplication bug already called out in the
   code. Touches only `ingestion/ingest.py`.
2. **Per-document grading in `grade_documents`.** Grade each retrieved
   document independently (one LLM call per doc, or one structured-output
   call returning a verdict per doc) and keep only the relevant subset,
   instead of keeping-all-or-dropping-all. This is the actual CRAG pattern and
   is a smaller conceptual fix than it sounds — same node, better prompt/loop.
3. **LangSmith tracing.** Set `LANGCHAIN_TRACING_V2=true` +
   `LANGCHAIN_API_KEY`. Zero code change, immediate visibility into every
   node's input/output — makes every other change below easier to debug.

### Tier 1 — retrieval quality (the actual "advanced RAG" work)

4. **Hybrid search (dense + sparse) via Qdrant's Query API + RRF.** Add a
   sparse vector (FastEmbed's BM25/SPLADE, ships with `qdrant-client`) at
   ingestion time, fuse with the existing dense search at query time. This is
   the single highest-value change for a docs corpus full of exact API names.
5. **Cross-encoder reranking.** Retrieve top ~20 via hybrid search, rerank
   down to 4 with a local cross-encoder (e.g. `BAAI/bge-reranker-base` via
   `sentence-transformers` or FastEmbed's reranker) before grading.
6. **Contextual retrieval at ingestion.** Generate a short context blurb per
   chunk with the LLM before embedding/BM25-indexing. Since ingestion already
   has LLM access configured, this is additive. Caveat: ~4000 extra LLM calls
   at ingestion time — fine with Ollama running locally overnight, would be
   costly against a paid API; consider capping/sampling if using
   `OPENAI_API_KEY`.

### Tier 2 — prove it worked

7. **A small evaluation harness.** 15-20 hand-written Q/A pairs about
   LangChain/LangGraph (things you actually know the answer to), scored with
   RAGAS (faithfulness, context precision/recall) or a simple custom
   LLM-judge script if RAGAS's dependency weight isn't worth it for a lab
   project. Run before/after each Tier 1 change to confirm it actually helped
   — otherwise "advanced" is just "different." Ideally this lands *before*
   Tier 1 so there's a baseline, but retrofitting is fine too.

### Tier 3 — UX, only if the chat experience itself is the next goal

8. **Streaming.** `astream_events` over the graph → SSE in FastAPI → streaming
   generator in Gradio. Currently fully blocking.
9. **Multi-turn memory.** The Gradio `ChatInterface` already receives
   `_history` and throws it away. Wire it into the graph state (or use a
   LangGraph checkpointer keyed by session) so follow-up questions work.

### Explicitly skipped (YAGNI at this scale)

- **GraphRAG / RAPTOR** — built for multi-hop reasoning over large,
  interconnected corpora. This corpus is flat reference docs; skip unless
  multi-hop questions become a real, observed pain point.
- **Semantic caching** — only pays off at query volumes this lab doesn't have.
- **Multi-query/HyDE query expansion** — the existing rewrite-and-retry loop
  already covers "the first query didn't work"; adding parallel query
  variants is real complexity for a 3B model whose rewrites are already a bit
  unreliable. Revisit only if Tier 0-2 don't close the gap on multi-hop
  questions.

## Sources

- [12 Advanced RAG Techniques: Beyond Naive Retrieval (2026)](https://atlan.com/know/advanced-rag-techniques/)
- [Advanced RAG — Hybrid Search, Reranking & Knowledge Graphs (2026)](https://myengineeringpath.dev/genai-engineer/advanced-rag/)
- [Advanced RAG techniques for high-performance LLM applications — Neo4j](https://neo4j.com/blog/genai/advanced-rag-techniques/)
- [Contextual Retrieval: Anthropic's Method for Cutting RAG Failures](https://medium.com/coinmonks/contextual-retrieval-anthropics-method-for-cutting-rag-failures-b28d98d57c48)
- [Anthropic's Contextual Retrieval: A Guide With Implementation — DataCamp](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic)
- [Agentic RAG: The 2026 Production Guide](https://www.marsdevs.com/guides/agentic-rag-2026-guide)
- [Agentic RAG in 2026: Patterns, Code, Observability](https://futureagi.com/blog/agentic-rag-systems-2025/)
- [RAG Evaluation Metrics: Best Practices — Patronus AI](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)
- [Master LLM Evaluation: RAGAS and LLM-as-Judge](https://letsdatascience.com/blog/llm-evaluation-ragas-llm-as-judge-and-production-evals)
- [Ragas docs](https://docs.ragas.io/en/stable/)
- [Hybrid Search with Reranking — Qdrant](https://qdrant.tech/documentation/tutorials-search-engineering/reranking-hybrid-search/)
- [Hybrid Search with Qdrant's Query API](https://qdrant.tech/articles/hybrid-search/)
- [Hybrid Search in RAG: Dense + Sparse, RRF — GoPenAI](https://blog.gopenai.com/hybrid-search-in-rag-dense-sparse-bm25-splade-reciprocal-rank-fusion-and-when-to-use-which-fafe4fd6156e)
- [When to use Graphs in RAG: A Comprehensive Analysis](https://arxiv.org/pdf/2506.05690)
- [RAG vs. GraphRAG: A Systematic Evaluation](https://arxiv.org/html/2502.11371v3)
- [Boosting RAG Efficiency with RAPTOR-Inspired Hierarchical Indexing](https://medium.com/@tam.tamanna18/boosting-rag-efficiency-with-raptor-inspired-hierarchical-indexing-for-scalable-retrieval-f3583312bd84)
- [Retrieval Is the Bottleneck: HyDE, Query Expansion, Multi-Query RAG](https://medium.com/@mudassar.hakim/retrieval-is-the-bottleneck-hyde-query-expansion-and-multi-query-rag-explained-for-production-c1842bed7f8a)
- [Dissecting Agentic RAG: Component Ablation for Multi-Hop QA with a Local 7B Model](https://arxiv.org/pdf/2606.21553)
- [Top LLM Observability Platforms in 2026 — MarkTechPost](https://www.marktechpost.com/2026/08/09/top-llm-observability-and-evaluation-platforms-in-2026-langfuse-langsmith-braintrust-arize-and-more-compared/)
- [LangSmith vs Langfuse (2026)](https://myengineeringpath.dev/tools/langsmith-vs-langfuse/)
