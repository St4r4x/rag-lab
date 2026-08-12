# ui/app.py
import gradio as gr

from ui.chat_tab import build_chat_tab
from ui.documents_tab import build_documents_tab
from ui.eval_tab import build_eval_tab

with gr.Blocks(title="rag-lab — LangChain/LangGraph doc assistant") as demo:
    with gr.Tab("Chat"):
        build_chat_tab()
    with gr.Tab("Évaluation"):
        eval_dropdown, load_runs = build_eval_tab()
    with gr.Tab("Documents"):
        build_documents_tab()

    demo.load(fn=load_runs, outputs=eval_dropdown)

if __name__ == "__main__":
    demo.launch()
