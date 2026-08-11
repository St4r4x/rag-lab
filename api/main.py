from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

from config import QDRANT_URL, get_llm, get_vectorstore
from graph.build import build_graph_v2

app = FastAPI(title="rag-lab")


@lru_cache
def get_graph():
    return build_graph_v2(get_llm(), get_vectorstore())


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    try:
        QdrantClient(url=QDRANT_URL).get_collections()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant unreachable: {exc}") from exc
    return {"status": "ok"}


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
