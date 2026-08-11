import subprocess
import tempfile
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import QDRANT_COLLECTION, QDRANT_URL, get_embeddings

DOCS_REPO_URL = "https://github.com/langchain-ai/docs.git"

# LangChain and LangGraph docs were consolidated into this single repo
# (langchain-ai/langchain and langchain-ai/langgraph no longer carry their
# own docs/ folder with markdown content). Verified 2026-08-11.
SOURCES = {
    "langchain": "src/oss/langchain",
    "langgraph": "src/oss/langgraph",
}

DOCS_BASE_URL = "https://docs.langchain.com/oss/python"


def build_doc_url(relative_path: str) -> str:
    trimmed = relative_path.removeprefix("src/oss/").removesuffix(".mdx").removesuffix(".md")
    return f"{DOCS_BASE_URL}/{trimmed}"


def clone_docs_repo(dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "--depth", "1", DOCS_REPO_URL, str(dest)],
        check=True,
        capture_output=True,
    )


def load_markdown_files(repo_dir: Path, source: str, subpath: str) -> list[dict]:
    docs_dir = repo_dir / subpath
    files = list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.mdx"))
    pages = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            pages.append(
                {
                    "text": text,
                    "source": source,
                    "path": str(f.relative_to(repo_dir)),
                    "url": build_doc_url(str(f.relative_to(repo_dir))),
                }
            )
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for page in pages:
        for chunk_text in splitter.split_text(page["text"]):
            chunks.append(
                {
                    "text": chunk_text,
                    "source": page["source"],
                    "path": page["path"],
                    "url": page["url"],
                }
            )
    return chunks


def main() -> None:
    embeddings = get_embeddings()
    all_chunks: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "docs"
        clone_docs_repo(dest)
        for source, subpath in SOURCES.items():
            pages = load_markdown_files(dest, source, subpath)
            if not pages:
                raise RuntimeError(
                    f"Source '{source}' (subpath '{subpath}') yielded 0 pages — "
                    "check SOURCES paths against the docs repo layout."
                )
            all_chunks.extend(chunk_pages(pages))

    # ponytail: from_texts() appends rather than upserting, so re-running this script
    # duplicates points instead of replacing them. Fine for a one-shot lab ingestion;
    # add a collection-recreate/upsert-by-id step if re-ingestion becomes routine.

    texts = [c["text"] for c in all_chunks]
    metadatas = [{"source": c["source"], "path": c["path"], "url": c["url"]} for c in all_chunks]

    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
    )
    print(f"Ingested {len(texts)} chunks into '{QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    main()
