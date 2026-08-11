from langgraph.graph import END, START, StateGraph

from graph.state import RAGState


def make_retrieve(vectorstore):
    def retrieve(state: RAGState) -> dict:
        docs = vectorstore.similarity_search(state["question"], k=4)
        return {"documents": docs}

    return retrieve


def make_generate(llm):
    def generate(state: RAGState) -> dict:
        if not state["documents"]:
            return {"generation": "Je n'ai pas assez d'information dans la documentation indexée pour répondre."}
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        response = llm.invoke(
            "Answer the question using only the context below. "
            "If the context is insufficient, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['question']}"
        )
        return {"generation": response.content}

    return generate


def build_graph_v1(llm, vectorstore):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve(vectorstore))
    graph.add_node("generate", make_generate(llm))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


MAX_RETRIES = 2


def make_grade_documents(llm):
    def grade_documents(state: RAGState) -> dict:
        docs_text = "\n\n".join(doc.page_content for doc in state["documents"])
        response = llm.invoke(
            "Answer strictly 'yes' or 'no'. Does the context below contain "
            "information relevant to the question, even partially? Answer "
            "'yes' unless the context is completely unrelated to the topic.\n\n"
            f"Question: {state['question']}\n\nContext:\n{docs_text}"
        )
        relevant = "yes" in response.content.strip().lower()
        return {"documents": state["documents"] if relevant else []}

    return grade_documents


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


def build_graph_v2(llm, vectorstore):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve(vectorstore))
    graph.add_node("grade_documents", make_grade_documents(llm))
    graph.add_node("rewrite_query", make_rewrite_query(llm))
    graph.add_node("generate", make_generate(llm))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()
