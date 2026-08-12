# ui/documents_tab.py
import os
from pathlib import Path

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def upload_document(file_path: str) -> str:
    path = Path(file_path)
    with path.open("rb") as f:
        response = httpx.post(f"{API_URL}/documents", files={"file": (path.name, f)}, timeout=60)
    response.raise_for_status()
    data = response.json()
    return f"{data['filename']} : {data['chunks_added']} chunk(s) ajouté(s)."


def build_documents_tab() -> None:
    file_input = gr.File(label="Fichier (.md ou .txt)", file_types=[".md", ".txt"])
    upload_button = gr.Button("Uploader")
    status = gr.Textbox(label="Résultat", interactive=False)

    upload_button.click(fn=upload_document, inputs=file_input, outputs=status)
