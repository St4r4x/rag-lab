# eval/run_eval.py
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from config import get_judge_llm, get_llm, get_reranker, get_vectorstore
from eval.judge import score_correctness, score_faithfulness
from graph.build import build_graph_v2

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def evaluate_one(graph, judge_llm, item: dict) -> dict:
    result = graph.invoke(
        {"question": item["question"], "documents": [], "generation": "", "retries": 0}
    )
    answer = result["generation"]
    context = "\n\n".join(doc.page_content for doc in result["documents"])
    sources = sorted({doc.metadata.get("url", "") for doc in result["documents"]})

    faithfulness = score_faithfulness(judge_llm, item["question"], context, answer)
    correctness = score_correctness(judge_llm, item["question"], item["reference_answer"], answer)

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": answer,
        "sources": sources,
        "faithfulness": faithfulness,
        "correctness": correctness,
    }


def write_report(results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out_path


def print_summary(results: list[dict]) -> None:
    faithfulness_scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    correctness_scores = [r["correctness"] for r in results if r["correctness"] is not None]
    print()
    if faithfulness_scores:
        print(
            f"Average faithfulness: {statistics.mean(faithfulness_scores):.2f} "
            f"({len(faithfulness_scores)}/{len(results)} scored)"
        )
    if correctness_scores:
        print(
            f"Average correctness: {statistics.mean(correctness_scores):.2f} "
            f"({len(correctness_scores)}/{len(results)} scored)"
        )


def main() -> None:
    llm = get_llm()
    judge_llm = get_judge_llm()
    graph = build_graph_v2(llm, get_vectorstore(), get_reranker())

    results = [evaluate_one(graph, judge_llm, item) for item in load_dataset()]
    for r in results:
        print(f"[{r['id']}] faithfulness={r['faithfulness']} correctness={r['correctness']} — {r['question']}")

    out_path = write_report(results)
    print(f"\nWrote {out_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
