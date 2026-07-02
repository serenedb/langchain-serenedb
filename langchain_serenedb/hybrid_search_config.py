"""Hybrid search configuration and fusion functions for SereneDBVectorStore.

The fusion functions (:func:`weighted_sum_ranking`, :func:`reciprocal_rank_fusion`) are
pure Python: they merge two already-ranked result lists (a dense vector branch and a
lexical BM25 branch) into a single ranking.

The lexical branch indexes the ``content`` column in an inverted index (with a text
search dictionary carrying ``frequency = true``) and scores matches with
``BM25(index.tableoid)``. See :class:`HybridSearchConfig`.
"""

from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from .indexes import DistanceStrategy

Row = Mapping[str, Any]


def _normalize_scores(
    results: Sequence[dict[str, Any]], is_distance_metric: bool
) -> Sequence[dict[str, Any]]:
    """Normalizes scores to a 0-1 scale, where 1 is best."""
    if not results:
        return []

    # Get scores from the last column of each result
    scores = [float(list(item.values())[-1]) for item in results]
    min_score, max_score = min(scores), max(scores)
    score_range = max_score - min_score

    if score_range == 0:
        # All documents are of the highest quality (1.0)
        for item in results:
            item["normalized_score"] = 1.0
        return list(results)

    for item in results:
        # Access the score again from the last column for calculation
        score = list(item.values())[-1]
        normalized = (score - min_score) / score_range
        if is_distance_metric:
            # For distance, a lower score is better, so we invert the result.
            item["normalized_score"] = 1.0 - normalized
        else:
            # For similarity (like keyword search), a higher score is better.
            item["normalized_score"] = normalized

    return list(results)


def weighted_sum_ranking(
    primary_search_results: Sequence[Row],
    secondary_search_results: Sequence[Row],
    primary_results_weight: float = 0.5,
    secondary_results_weight: float = 0.5,
    fetch_top_k: int = 4,
    **kwargs: Any,
) -> Sequence[dict[str, Any]]:
    """Ranks documents using a weighted sum of scores from two sources.

    Args:
        primary_search_results: A list of (document, distance) tuples from
            the primary (vector) search.
        secondary_search_results: A list of (document, distance) tuples from
            the secondary (keyword/BM25) search.
        primary_results_weight: The weight for the primary source's scores.
        secondary_results_weight: The weight for the secondary source's scores.
        fetch_top_k: The number of documents to fetch after merging the results.

    Returns:
        A list of rows, sorted by weighted score in descending order.
    """
    distance_strategy = kwargs.get(
        "distance_strategy", DistanceStrategy.COSINE_DISTANCE
    )
    is_primary_distance = distance_strategy != DistanceStrategy.INNER_PRODUCT

    # Normalize both sets of results onto a 0-1 scale
    normalized_primary = _normalize_scores(
        [dict(row) for row in primary_search_results],
        is_distance_metric=is_primary_distance,
    )

    # Keyword search relevance is a similarity score (higher is better)
    normalized_secondary = _normalize_scores(
        [dict(row) for row in secondary_search_results], is_distance_metric=False
    )

    # stores computed metric with provided distance metric and weights
    weighted_scores: dict[str, dict[str, Any]] = {}

    # Process primary results
    for item in normalized_primary:
        doc_id = str(list(item.values())[0])
        item["distance"] = item["normalized_score"] * primary_results_weight
        weighted_scores[doc_id] = item

    # Process secondary results
    for item in normalized_secondary:
        doc_id = str(list(item.values())[0])
        secondary_weighted_score = item["normalized_score"] * secondary_results_weight

        if doc_id in weighted_scores:
            weighted_scores[doc_id]["distance"] += secondary_weighted_score
        else:
            item["distance"] = secondary_weighted_score
            weighted_scores[doc_id] = item

    ranked_results = sorted(
        weighted_scores.values(), key=lambda item: item["distance"], reverse=True
    )

    for result in ranked_results:
        result.pop("normalized_score", None)

    return ranked_results[:fetch_top_k]


def reciprocal_rank_fusion(
    primary_search_results: Sequence[Row],
    secondary_search_results: Sequence[Row],
    rrf_k: float = 60,
    fetch_top_k: int = 4,
    **kwargs: Any,
) -> Sequence[dict[str, Any]]:
    """Ranks documents using Reciprocal Rank Fusion (RRF) of two sources.

    Args:
        primary_search_results: A list of (document, distance) tuples from
            the primary (vector) search.
        secondary_search_results: A list of (document, distance) tuples from
            the secondary (keyword/BM25) search.
        rrf_k: The RRF parameter k.
        fetch_top_k: The number of documents to fetch after merging the results.

    Returns:
        A list of rows, sorted by RRF score in descending order.
    """
    rrf_scores: dict[str, dict[str, Any]] = {}

    # Determine sorting order based on the vector distance strategy.
    # For COSINE & EUCLIDEAN (distance), we sort ascending (reverse=False).
    # For INNER_PRODUCT (similarity), we sort descending (reverse=True).
    distance_strategy = kwargs.get(
        "distance_strategy", DistanceStrategy.COSINE_DISTANCE
    )
    is_similarity_metric = distance_strategy == DistanceStrategy.INNER_PRODUCT
    sorted_primary = sorted(
        primary_search_results,
        key=lambda item: item["distance"],
        reverse=is_similarity_metric,
    )

    for rank, row in enumerate(sorted_primary):
        doc_id = str(list(row.values())[0])
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = dict(row)
            rrf_scores[doc_id]["distance"] = 0.0
        rrf_scores[doc_id]["distance"] += 1.0 / (rank + rrf_k)

    # Keyword search relevance is always "higher is better" -> sort descending
    sorted_secondary = sorted(
        secondary_search_results,
        key=lambda item: item["distance"],
        reverse=True,
    )

    for rank, row in enumerate(sorted_secondary):
        doc_id = str(list(row.values())[0])
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = dict(row)
            rrf_scores[doc_id]["distance"] = 0.0
        rrf_scores[doc_id]["distance"] += 1.0 / (rank + rrf_k)

    ranked_results = sorted(
        rrf_scores.values(), key=lambda item: item["distance"], reverse=True
    )
    return ranked_results[:fetch_top_k]


@dataclass
class HybridSearchConfig(ABC):
    """SereneDB vector store hybrid-search configuration.

    The lexical branch is served by an inverted index on the ``content`` column.
    Scoring requires the column's text search dictionary to carry ``frequency = true``
    (and ``position = true`` for phrase queries, ``norm = true`` for the language-model
    scorers). Because the ``@@`` predicate resolves only when selecting *from the index
    by name*, the store also stores the id/metadata columns via ``INCLUDE`` so the
    lexical branch can return them.

    Attributes:
        fts_query: The full-text query string. If empty, the search query text is used.
        fusion_function: Callable that merges the vector and keyword result lists.
            Defaults to :func:`weighted_sum_ranking`.
        fusion_function_parameters: Extra kwargs forwarded to ``fusion_function``.
        primary_top_k: Rows to fetch from the vector branch before fusion.
        secondary_top_k: Rows to fetch from the keyword branch before fusion.
        index_name: Name of the inverted index that covers the content column.
        dictionary_name: Text search dictionary used to analyze the content column.
        dictionary_options: Options for ``CREATE TEXT SEARCH DICTIONARY`` — must include
            ``frequency = true`` for scoring.
        scorer: Relevance scorer function name (``BM25``, ``TFIDF``, ``dfi``, ...).
        tsquery_function: Query constructor (``plainto_tsquery``, ``to_tsquery``,
            ``phraseto_tsquery`` or ``websearch_to_tsquery``).
    """

    fts_query: str = ""
    fusion_function: Callable[
        [Sequence[Row], Sequence[Row], Any], Sequence[Any]
    ] = weighted_sum_ranking
    fusion_function_parameters: dict[str, Any] = field(default_factory=dict)
    primary_top_k: int = 4
    secondary_top_k: int = 4
    index_name: str = "langchain_inverted_index"
    dictionary_name: str = "langchain_fts_dict"
    dictionary_options: str = (
        "template = 'text', locale = 'en_US.UTF-8', "
        "frequency = true, position = true, norm = true"
    )
    scorer: str = "BM25"
    tsquery_function: str = "plainto_tsquery"
