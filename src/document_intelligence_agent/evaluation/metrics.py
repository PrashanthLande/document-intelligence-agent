"""Ranking metrics over the rank of each question's first relevant result.

A rank is 1-based; None means no relevant result was returned.
"""


def first_relevant_rank(relevance: list[bool]) -> int | None:
    for index, relevant in enumerate(relevance, start=1):
        if relevant:
            return index

    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions with a relevant result in the top k."""
    if not ranks:
        return 0.0

    hits = sum(1 for rank in ranks if rank is not None and rank <= k)

    return hits / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Average of 1/rank, counting 0 for questions with no relevant result."""
    if not ranks:
        return 0.0

    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)
