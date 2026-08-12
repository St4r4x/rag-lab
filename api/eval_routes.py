# api/eval_routes.py
import json
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from eval.run_eval import RESULTS_DIR, summarize

router = APIRouter(prefix="/eval", tags=["eval"])

RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")


class EvalRunSummary(BaseModel):
    id: str
    count: int
    avg_faithfulness: float | None
    avg_correctness: float | None


def _summary_response(run_id: str, results: list[dict]) -> EvalRunSummary:
    summary = summarize(results)
    return EvalRunSummary(
        id=run_id,
        count=summary["count"],
        avg_faithfulness=summary["avg_faithfulness"],
        avg_correctness=summary["avg_correctness"],
    )


@router.get("/runs", response_model=list[EvalRunSummary])
def list_eval_runs():
    if not RESULTS_DIR.exists():
        return []
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        results = json.loads(path.read_text(encoding="utf-8"))
        runs.append(_summary_response(path.stem, results))
    return runs


@router.get("/runs/{run_id}")
def get_eval_run(run_id: str):
    if not RUN_ID_PATTERN.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run id")
    path = RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return json.loads(path.read_text(encoding="utf-8"))
