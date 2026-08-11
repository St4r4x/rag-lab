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
