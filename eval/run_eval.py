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


def summarize(results: list[dict]) -> dict:
    faithfulness_scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    correctness_scores = [r["correctness"] for r in results if r["correctness"] is not None]
    return {
        "count": len(results),
        "avg_faithfulness": statistics.mean(faithfulness_scores) if faithfulness_scores else None,
        "faithfulness_scored": len(faithfulness_scores),
        "avg_correctness": statistics.mean(correctness_scores) if correctness_scores else None,
        "correctness_scored": len(correctness_scores),
    }


def print_summary(results: list[dict]) -> None:
    summary = summarize(results)
    print()
    if summary["avg_faithfulness"] is not None:
        print(
            f"Average faithfulness: {summary['avg_faithfulness']:.2f} "
            f"({summary['faithfulness_scored']}/{summary['count']} scored)"
        )
    if summary["avg_correctness"] is not None:
        print(
            f"Average correctness: {summary['avg_correctness']:.2f} "
            f"({summary['correctness_scored']}/{summary['count']} scored)"
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
