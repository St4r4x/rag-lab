from langgraph.graph import END, START, StateGraph
from langchain_core.documents import Document

from graph.state import RAGState


def make_retrieve(vectorstore, k: int):
    def retrieve(state: RAGState) -> dict:
        docs = vectorstore.similarity_search(state["question"], k=k)
        return {"documents": docs}

    return retrieve


def make_generate(llm):
    def generate(state: RAGState) -> dict:
        if not state["documents"]:
            return {"generation": "Je n'ai pas assez d'information dans la documentation indexée pour répondre."}
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        response = llm.invoke(
            "You are answering a question using excerpts from official documentation. "
            "Synthesize a clear answer from the context below, extracting and explaining "
            "whatever relevant information it contains, even if partial. Only say the "
            "context is insufficient if it is truly unrelated to the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['question']}"
        )
        return {"generation": response.content}

    return generate


MAX_RETRIES = 2
RETRIEVE_POOL_SIZE = 20
RERANK_TOP_K = 4


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


def route_after_grade(state: RAGState) -> str:
    if state["documents"]:
        return "generate"
    if state["retries"] >= MAX_RETRIES:
        return "generate"
    return "rewrite_query"


def make_rewrite_query(llm):
    def rewrite_query(state: RAGState) -> dict:
        response = llm.invoke(
            "Reformulate this question to improve document retrieval. "
            f"Keep it concise, return only the reformulated question:\n{state['question']}"
        )
        return {"question": response.content.strip(), "retries": state["retries"] + 1}

    return rewrite_query


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
