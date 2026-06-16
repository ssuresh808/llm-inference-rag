"""Deterministic retrieval-quality metrics.

These operate on ranked lists of source identifiers and need no LLM, so they are
fast, reproducible, and unit-testable. Used by the retrieval evaluation harness.
"""

from collections.abc import Sequence


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return 1.0 if any relevant source is in the top-k retrieved, else 0.0.

    Args:
        retrieved: Ranked source identifiers (best first).
        relevant: The set of relevant source identifiers.
        k: Cut-off rank.

    Returns:
        1.0 on a hit within the top k, otherwise 0.0.
    """
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Return the reciprocal of the rank of the first relevant source.

    Args:
        retrieved: Ranked source identifiers (best first).
        relevant: The set of relevant source identifiers.

    Returns:
        ``1 / rank`` of the first relevant hit, or 0.0 if none are relevant.
    """
    for index, source in enumerate(retrieved, start=1):
        if source in relevant:
            return 1.0 / index
    return 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Return the fraction of relevant sources found in the top-k retrieved.

    Args:
        retrieved: Ranked source identifiers (best first).
        relevant: The set of relevant source identifiers.
        k: Cut-off rank.

    Returns:
        ``|relevant intersect top-k| / |relevant|``, or 0.0 when there are no
        relevant sources.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)
