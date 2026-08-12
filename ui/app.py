# ui/app.py
import gradio as gr

from ui.chat_tab import build_chat_tab

with gr.Blocks(title="rag-lab — LangChain/LangGraph doc assistant") as demo:
    with gr.Tab("Chat"):
        build_chat_tab()

if __name__ == "__main__":
    demo.launch()
