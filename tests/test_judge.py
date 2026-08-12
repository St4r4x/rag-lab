# tests/test_judge.py
from types import SimpleNamespace

from eval.judge import score_correctness, score_faithfulness


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, _prompt):
        return SimpleNamespace(content=self.responses.pop(0))


def test_score_faithfulness_parses_digit_from_response():
    llm = FakeLLM(["4"])
    score = score_faithfulness(llm, "q", "context text", "answer text")
    assert score == 4


def test_score_faithfulness_returns_none_on_unparseable_response():
    llm = FakeLLM(["I cannot determine this."])
    score = score_faithfulness(llm, "q", "context text", "answer text")
    assert score is None


def test_score_correctness_parses_digit_from_response():
    llm = FakeLLM(["5"])
    score = score_correctness(llm, "q", "reference answer", "generated answer")
    assert score == 5


def test_score_correctness_extracts_digit_from_verbose_response():
    llm = FakeLLM(["I would rate this a 3 out of 5."])
    score = score_correctness(llm, "q", "reference answer", "generated answer")
    assert score == 3
