# tests/test_eval_tab.py
from ui.eval_tab import build_dropdown_choices, detail_to_rows, format_run_label


def test_format_run_label_with_scores():
    run = {"id": "20260101T000000Z", "count": 18, "avg_faithfulness": 3.9, "avg_correctness": 4.0}
    label = format_run_label(run)
    assert "20260101T000000Z" in label
    assert "18" in label
    assert "3.90" in label
    assert "4.00" in label


def test_format_run_label_handles_unscored():
    run = {"id": "r1", "count": 2, "avg_faithfulness": None, "avg_correctness": None}
    label = format_run_label(run)
    assert "n/a" in label


def test_build_dropdown_choices():
    runs = [{"id": "r1", "count": 1, "avg_faithfulness": 4.0, "avg_correctness": 5.0}]
    choices = build_dropdown_choices(runs)
    assert choices == [(format_run_label(runs[0]), "r1")]


def test_detail_to_rows():
    detail = [{"id": "q01", "category": "identifier", "question": "What?", "faithfulness": 4, "correctness": 5}]
    rows = detail_to_rows(detail)
    assert rows == [["q01", "identifier", "What?", 4, 5]]
