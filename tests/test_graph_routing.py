# tests/test_graph_routing.py
from types import SimpleNamespace

from langchain_core.documents import Document

from graph.build import (
    MAX_RETRIES,
    build_graph_v2,
    make_grade_documents,
    make_rerank,
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


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def rerank(self, _query, _documents):
        return self.scores


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


def test_grade_documents_keeps_only_relevant_ones():
    grade = make_grade_documents(FakeLLM(["yes", "no"]))
    relevant_doc = Document(page_content="relevant text")
    irrelevant_doc = Document(page_content="irrelevant text")
    state = {"question": "q", "documents": [relevant_doc, irrelevant_doc]}
    result = grade(state)
    assert result["documents"] == [relevant_doc]


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


def test_graph_v2_falls_back_after_max_retries_without_infinite_loop():
    vectorstore = FakeVectorStore(docs=[Document(page_content="irrelevant")])
    reranker = FakeReranker(scores=[1.0])
    llm = FakeLLM(responses=["no", "reformulated question 1", "no", "reformulated question 2", "no"])
    graph = build_graph_v2(llm, vectorstore, reranker)

    result = graph.invoke({"question": "what is x", "documents": [], "generation": "", "retries": 0})

    assert result["generation"] == "Je n'ai pas assez d'information dans la documentation indexée pour répondre."
    assert vectorstore.search_calls == 3
