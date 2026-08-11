from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    retries: int
