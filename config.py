import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "langchain_docs")


class _NomicPrefixedEmbeddings(Embeddings):
    """ponytail: nomic-embed-text needs search_document:/search_query: prefixes
    for good asymmetric retrieval (per its model card). Only wraps when that
    exact model is selected; other EMBEDDING_MODEL values pass through
    untouched from init_embeddings()."""

    def __init__(self, inner: Embeddings):
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(f"search_query: {text}")


def get_llm():
    model = os.environ.get("LLM_MODEL", "ollama:llama3.2:3b")
    return init_chat_model(model)


def get_embeddings():
    model = os.environ.get("EMBEDDING_MODEL", "ollama:nomic-embed-text")
    embeddings = init_embeddings(model)
    if "nomic-embed-text" in model:
        return _NomicPrefixedEmbeddings(embeddings)
    return embeddings


@lru_cache
def get_vectorstore() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=get_embeddings(),
    )
