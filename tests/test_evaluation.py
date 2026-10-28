from pathlib import Path

from document_intelligence_agent.evaluation.dataset import (
    EvalQuestion,
    is_relevant,
    load_dataset,
)
from document_intelligence_agent.evaluation.metrics import (
    first_relevant_rank,
    hit_at_k,
    mean_reciprocal_rank,
)

QUESTION = EvalQuestion(
    id="q",
    question="How long does a table take?",
    document_id="report.pdf",
    pages=[4],
    evidence=["between 2 and 6 seconds"],
)


def test_first_relevant_rank_is_one_based():
    assert first_relevant_rank([False, True, True]) == 2
    assert first_relevant_rank([False, False]) is None


def test_hit_at_k_and_mrr():
    ranks = [1, 3, None, 2]

    assert hit_at_k(ranks, 1) == 0.25
    assert hit_at_k(ranks, 3) == 0.75
    assert mean_reciprocal_rank(ranks) == (1 + 1 / 3 + 0 + 1 / 2) / 4


def test_metrics_of_no_questions_are_zero():
    assert hit_at_k([], 3) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


def test_relevant_chunk_ignores_case_and_line_breaks():
    text = "Typical tables require Between 2 and\n6 seconds on a CPU."

    assert is_relevant(QUESTION, "report.pdf", [4], text)


def test_chunk_on_wrong_page_or_document_is_not_relevant():
    text = "between 2 and 6 seconds"

    assert not is_relevant(QUESTION, "report.pdf", [5], text)
    assert not is_relevant(QUESTION, "other.pdf", [4], text)
    assert not is_relevant(QUESTION, "report.pdf", [4], "no answer here")


def test_dataset_file_is_valid():
    dataset = load_dataset(Path("eval/docling_technical_report.json"))
    document_ids = {doc.document_id for doc in dataset.documents}

    assert len(dataset.questions) == 20
    assert len({q.id for q in dataset.questions}) == 20

    for question in dataset.questions:
        assert question.document_id in document_ids
        assert question.pages
        assert question.evidence
