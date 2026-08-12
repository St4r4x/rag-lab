# api/documents_routes.py
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from config import get_vectorstore
from ingestion.ingest import chunk_document

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_SUFFIXES = {".md", ".txt"}
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_added: int


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile) -> DocumentUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or '(none)'}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 5 MB)")

    text = content.decode("utf-8", errors="ignore")
    chunks = chunk_document(text, source="upload", path=file.filename, url="")
    if not chunks:
        raise HTTPException(status_code=400, detail="No content extracted from file")

    try:
        get_vectorstore().add_texts(
            texts=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "path": c["path"], "url": c["url"]} for c in chunks],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return DocumentUploadResponse(filename=file.filename, chunks_added=len(chunks))
