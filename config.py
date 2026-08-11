import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "langchain_docs")


def get_llm():
    model = os.environ.get("LLM_MODEL", "ollama:llama3.2:3b")
    return init_chat_model(model)


def get_embeddings():
    model = os.environ.get("EMBEDDING_MODEL", "ollama:nomic-embed-text")
    return init_embeddings(model)


@lru_cache
def get_vectorstore() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=get_embeddings(),
    )
