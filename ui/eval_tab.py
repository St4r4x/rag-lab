# ui/eval_tab.py
import os

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def fetch_runs() -> list[dict]:
    response = httpx.get(f"{API_URL}/eval/runs", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_run_detail(run_id: str) -> list[dict]:
    response = httpx.get(f"{API_URL}/eval/runs/{run_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def trigger_run() -> list[dict]:
    response = httpx.post(f"{API_URL}/eval/run", timeout=600)
    response.raise_for_status()
    return fetch_runs()


def format_run_label(run: dict) -> str:
    faithfulness = f"{run['avg_faithfulness']:.2f}" if run["avg_faithfulness"] is not None else "n/a"
    correctness = f"{run['avg_correctness']:.2f}" if run["avg_correctness"] is not None else "n/a"
    return f"{run['id']} — {run['count']} questions, faithfulness {faithfulness}, correctness {correctness}"


def build_dropdown_choices(runs: list[dict]) -> list[tuple[str, str]]:
    return [(format_run_label(run), run["id"]) for run in runs]


def detail_to_rows(detail: list[dict]) -> list[list]:
    return [
        [item["id"], item["category"], item["question"], item["faithfulness"], item["correctness"]]
        for item in detail
    ]


def build_eval_tab() -> tuple[gr.Dropdown, callable]:
    dropdown = gr.Dropdown(choices=[], label="Run")
    table = gr.Dataframe(
        headers=["id", "category", "question", "faithfulness", "correctness"],
        label="Détail",
    )
    refresh_button = gr.Button("Rafraîchir la liste")
    run_button = gr.Button("Lancer une évaluation")

    def on_select(run_id):
        if not run_id:
            return []
        return detail_to_rows(fetch_run_detail(run_id))

    def load_runs():
        runs = fetch_runs()
        return gr.Dropdown(choices=build_dropdown_choices(runs), value=runs[0]["id"] if runs else None)

    def on_run():
        runs = trigger_run()
        return gr.Dropdown(choices=build_dropdown_choices(runs), value=runs[0]["id"] if runs else None)

    dropdown.change(fn=on_select, inputs=dropdown, outputs=table)
    refresh_button.click(fn=load_runs, outputs=dropdown)
    run_button.click(fn=on_run, outputs=dropdown)

    return dropdown, load_runs
