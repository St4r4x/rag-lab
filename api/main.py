# api/main.py
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

from api.dependencies import get_graph
from config import QDRANT_COLLECTION, QDRANT_URL

app = FastAPI(title="rag-lab")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class ConfigResponse(BaseModel):
    llm_model: str
    embedding_model: str
    sparse_embedding_model: str
    reranker_model: str
    judge_model: str
    qdrant_url: str
    qdrant_collection: str


@app.get("/health")
def health():
    try:
        QdrantClient(url=QDRANT_URL).get_collections()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant unreachable: {exc}") from exc
    return {"status": "ok"}


@app.get("/config", response_model=ConfigResponse)
def get_config():
    llm_model = os.environ.get("LLM_MODEL", "ollama:llama3.2:3b")
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "") or llm_model
    return ConfigResponse(
        llm_model=llm_model,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "ollama:nomic-embed-text"),
        sparse_embedding_model=os.environ.get("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25"),
        reranker_model=os.environ.get("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        judge_model=judge_model,
        qdrant_url=QDRANT_URL,
        qdrant_collection=QDRANT_COLLECTION,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        result = get_graph().invoke(
            {"question": request.question, "documents": [], "generation": "", "retries": 0}
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    sources = sorted({doc.metadata.get("url", "") for doc in result["documents"]})
    return QueryResponse(answer=result["generation"], sources=sources)
