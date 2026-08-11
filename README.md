# rag-lab

Projet d'apprentissage : un RAG (Retrieval-Augmented Generation) construit avec LangChain et LangGraph, qui répond à des questions sur la documentation officielle de ces deux frameworks.

Le pipeline part d'un graphe simple (`retrieve → generate`) et évolue — sur le même `StateGraph` — vers une version agentique qui juge la pertinence des documents récupérés et reformule la question en cas d'échec (boucle bornée à 2 tentatives) avant de répondre.

## Stack

- **Orchestration** : LangGraph (`StateGraph`)
- **LLM / embeddings** : LangChain (`init_chat_model` / `init_embeddings`), switchable Ollama (local) ↔ API via variables d'environnement
- **Vector store** : Qdrant (Docker)
- **API** : FastAPI
- **UI** : Gradio

## Quickstart

Prérequis : Docker, [Ollama](https://ollama.com) installé et lancé (`ollama serve`).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

cp .env.example .env   # ajuster si besoin (provider, modèles, ports)

ollama pull llama3.2:3b
ollama pull nomic-embed-text

docker compose up -d                    # Qdrant sur localhost:6333
.venv/bin/python -m ingestion.ingest     # indexe la doc LangChain/LangGraph (~4000 chunks, quelques minutes)

.venv/bin/uvicorn api.main:app --port 8000 &
.venv/bin/python -m ui.app                # UI de chat sur http://127.0.0.1:7860
```

## Configuration

Variables d'environnement (voir `.env.example`) :

| Variable | Défaut | Description |
|---|---|---|
| `LLM_MODEL` | `ollama:llama3.2:3b` | Modèle de chat (`init_chat_model`) |
| `EMBEDDING_MODEL` | `ollama:nomic-embed-text` | Modèle d'embeddings (`init_embeddings`) |
| `QDRANT_URL` | `http://localhost:6333` | URL Qdrant |
| `QDRANT_COLLECTION` | `langchain_docs` | Collection Qdrant |
| `RAG_API_URL` | `http://localhost:8000` | URL de l'API, utilisée par l'UI |

Pour utiliser une API au lieu d'Ollama : `LLM_MODEL=openai:gpt-4o-mini`, `EMBEDDING_MODEL=openai:text-embedding-3-small`, `OPENAI_API_KEY=...`.

## Structure

```
ingestion/   # clone + chunk + embed + upsert la doc LangChain/LangGraph
graph/       # StateGraph : v1 (retrieve→generate) → v2 (+ grade/rewrite/loop)
config.py    # LLM / embeddings / vector store
api/         # FastAPI (POST /query, GET /health)
ui/          # UI de chat Gradio
tests/       # test du routing du graphe v2 (seule suite automatisée, décision volontaire)
```

Design complet : [docs/superpowers/specs/2026-08-11-rag-lab-design.md](docs/superpowers/specs/2026-08-11-rag-lab-design.md)
