# ui/config_tab.py
import os

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

CONFIG_LABELS = {
    "llm_model": "Modèle LLM",
    "embedding_model": "Modèle d'embeddings",
    "sparse_embedding_model": "Modèle sparse",
    "reranker_model": "Modèle de reranking",
    "judge_model": "Modèle juge (éval)",
    "qdrant_url": "URL Qdrant",
    "qdrant_collection": "Collection Qdrant",
}


def fetch_config() -> dict:
    response = httpx.get(f"{API_URL}/config", timeout=30)
    response.raise_for_status()
    return response.json()


def format_config_rows(config: dict) -> list[list[str]]:
    return [[CONFIG_LABELS.get(key, key), value] for key, value in config.items()]


def build_config_tab() -> tuple[gr.Dataframe, callable]:
    table = gr.Dataframe(headers=["Paramètre", "Valeur"], label="Configuration actuelle")
    refresh_button = gr.Button("Rafraîchir")

    def load_config():
        return format_config_rows(fetch_config())

    refresh_button.click(fn=load_config, outputs=table)
    return table, load_config
