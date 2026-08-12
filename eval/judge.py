# eval/judge.py
import re


def _parse_score(text: str) -> int | None:
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else None


def score_faithfulness(llm, question: str, context: str, answer: str) -> int | None:
    response = llm.invoke(
        "You are evaluating whether an answer is faithful to (i.e., supported by) the given "
        "context. Rate faithfulness from 1 (answer is unsupported or contradicts the context) "
        "to 5 (answer is fully supported by the context). Respond with only a single digit "
        "from 1 to 5.\n\n"
        f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
    )
    return _parse_score(response.content)


def score_correctness(llm, question: str, reference_answer: str, answer: str) -> int | None:
    response = llm.invoke(
        "You are evaluating whether a generated answer correctly addresses a question, "
        "compared to a reference answer. Rate correctness from 1 (wrong or irrelevant) to "
        "5 (fully correct and complete). Respond with only a single digit from 1 to 5.\n\n"
        f"Question:\n{question}\n\nReference answer:\n{reference_answer}\n\n"
        f"Generated answer:\n{answer}"
    )
    return _parse_score(response.content)
