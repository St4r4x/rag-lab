# api/dependencies.py
from functools import lru_cache

from config import get_llm, get_reranker, get_vectorstore
from graph.build import build_graph_v2


@lru_cache
def get_graph():
    return build_graph_v2(get_llm(), get_vectorstore(), get_reranker())
