# tests/test_run_eval.py
from eval.run_eval import summarize


def test_summarize_all_scored():
    results = [
        {"faithfulness": 4, "correctness": 5},
        {"faithfulness": 2, "correctness": 3},
    ]
    summary = summarize(results)
    assert summary == {
        "count": 2,
        "avg_faithfulness": 3.0,
        "faithfulness_scored": 2,
        "avg_correctness": 4.0,
        "correctness_scored": 2,
    }


def test_summarize_partially_scored():
    results = [
        {"faithfulness": 4, "correctness": None},
        {"faithfulness": None, "correctness": 3},
    ]
    summary = summarize(results)
    assert summary["count"] == 2
    assert summary["avg_faithfulness"] == 4.0
    assert summary["faithfulness_scored"] == 1
    assert summary["avg_correctness"] == 3.0
    assert summary["correctness_scored"] == 1


def test_summarize_none_scored():
    results = [{"faithfulness": None, "correctness": None}]
    summary = summarize(results)
    assert summary["count"] == 1
    assert summary["avg_faithfulness"] is None
    assert summary["faithfulness_scored"] == 0
    assert summary["avg_correctness"] is None
    assert summary["correctness_scored"] == 0
