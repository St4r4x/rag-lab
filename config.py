import os
from functools import lru_cache

from dotenv import load_dotenv
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "langchain_docs")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "")
SPARSE_EMBEDDING_MODEL = os.environ.get("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")


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


def _ollama_kwargs(model: str) -> dict:
    if OLLAMA_BASE_URL and model.startswith("ollama:"):
        return {"base_url": OLLAMA_BASE_URL}
    return {}


def get_llm():
    model = os.environ.get("LLM_MODEL", "ollama:llama3.2:3b")
    return init_chat_model(model, **_ollama_kwargs(model))


def get_judge_llm():
    model = os.environ.get("EVAL_JUDGE_MODEL", "")
    if not model:
        return get_llm()
    return init_chat_model(model, **_ollama_kwargs(model))


def get_embeddings():
    model = os.environ.get("EMBEDDING_MODEL", "ollama:nomic-embed-text")
    embeddings = init_embeddings(model, **_ollama_kwargs(model))
    if "nomic-embed-text" in model:
        return _NomicPrefixedEmbeddings(embeddings)
    return embeddings


def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL)


@lru_cache
def get_vectorstore() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
    )


@lru_cache
def get_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANKER_MODEL)
