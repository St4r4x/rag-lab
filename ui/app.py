import os

import gradio as gr
import httpx

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def ask(message: str, _history) -> str:
    response = httpx.post(f"{API_URL}/query", json={"question": message}, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not data["sources"]:
        return data["answer"]
    sources = "\n".join(f"- {s}" for s in data["sources"])
    return f"{data['answer']}\n\nSources:\n{sources}"


demo = gr.ChatInterface(fn=ask, title="rag-lab — LangChain/LangGraph doc assistant")

if __name__ == "__main__":
    demo.launch()
