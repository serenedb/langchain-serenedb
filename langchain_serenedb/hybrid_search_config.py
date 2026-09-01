"""Hybrid search configuration for SereneDBVectorStore.

Hybrid search fuses a dense (vector ANN) ranking and a lexical (BM25 full-text) ranking
in a **single** SereneDB query — a ``WITH fused AS (lexical UNION ALL vector)`` CTE whose
outer query combines the two per the chosen :class:`FusionStrategy`. The lexical branch
indexes the ``content`` column in an inverted index (with a text search dictionary
carrying ``frequency = true``) and scores matches with ``BM25(index.tableoid)``.

The configuration is split by concern:

- :class:`HybridIndexConfig` — the *build-time* side (the text search dictionary). Passed
  to ``init_vectorstore_table`` / ``apply_hybrid_search_index`` when creating the index.
- :class:`HybridSearchConfig` — the *query-time* side (fusion strategy, weights, windows,
  scorer, tsquery function). Held by the store and applied to each search.
"""

import enum
from dataclasses import dataclass


class FusionStrategy(str, enum.Enum):
    """How the lexical (BM25) and vector rankings are combined into one score.

    - ``RRF`` (the default): Reciprocal Rank Fusion — combine *ranks*, not scores, with
      ``sum(1 / (rrf_k + rank))``. Scale-free, so BM25 and vector distance fuse as-is
      with no per-branch weights. Choose it when consensus should win: a document several
      signals agree on beats one only a single signal likes.
    - ``NORMALIZED``: min-max normalize each branch's scores to ``[0, 1]`` (the vector
      distance branch is inverted so nearer = higher), then take a weighted sum. Keeps
      score margins, so a decisive win in one branch can outrank lukewarm presence in
      several; the trade-off is that one outlier rescales its whole branch.
    - ``WEIGHTED_SUM``: a plain weighted sum of the *raw* branch scores
      (``primary_results_weight * vector + secondary_results_weight * bm25``). Keeps
      magnitudes with no min-max distortion, but is only meaningful when the branches are
      already on comparable scales — BM25 and a raw vector distance usually are not.
    """

    RRF = "rrf"
    NORMALIZED = "normalized"
    WEIGHTED_SUM = "weighted_sum"


@dataclass
class HybridIndexConfig:
    """Build-time configuration for the hybrid (combined full-text + vector) index.

    Describes the text search dictionary the ``content`` column is analyzed with when the
    combined inverted index is created. Used only while *building* the index (by
    ``init_vectorstore_table`` and ``apply_hybrid_search_index``); it is not needed at
    query time (the ``@@`` predicate resolves through the index's own analyzer).

    Attributes:
        dictionary_name: Name of the text search dictionary created for the content column.
        dictionary_options: Options for ``CREATE TEXT SEARCH DICTIONARY`` — must include
            ``frequency = true`` for scoring (and ``position = true`` for phrase queries,
            ``norm = true`` for the language-model scorers).
    """

    dictionary_name: str = "langchain_fts_dict"
    dictionary_options: str = (
        "template = 'segmentation', case = 'lower', "
        "frequency = true, position = true, norm = true"
    )


@dataclass
class HybridSearchConfig:
    """Query-time configuration for hybrid search.

    Held by the store and applied to each hybrid query: how the lexical and vector
    branches are windowed, scored, and fused. It carries no index/dictionary settings
    (see :class:`HybridIndexConfig` for those) — the index must already exist.

    The full-text query is per-call, not stored here: the store uses the search query
    text (or an explicit ``fts_query`` passed to ``similarity_search``).

    Attributes:
        fusion: How the two rankings are combined (see :class:`FusionStrategy`). Defaults
            to :attr:`FusionStrategy.RRF`.
        rrf_k: The RRF ``k`` constant (used only when ``fusion`` is ``RRF``). Higher ``k``
            flattens the weight of top ranks; ``60`` is the published default.
        primary_results_weight: Weight of the vector branch for the ``NORMALIZED`` and
            ``WEIGHTED_SUM`` strategies.
        secondary_results_weight: Weight of the lexical (BM25) branch for the
            ``NORMALIZED`` and ``WEIGHTED_SUM`` strategies.
        primary_top_k: Per-branch window — rows the vector branch contributes to fusion.
        secondary_top_k: Per-branch window — rows the keyword branch contributes.
        scorer: Relevance scorer function name (``BM25``, ``TFIDF``, ``dfi``, ...).
        tsquery_function: Query constructor (``plainto_tsquery``, ``to_tsquery``,
            ``phraseto_tsquery`` or ``websearch_to_tsquery``).
    """

    fusion: FusionStrategy = FusionStrategy.RRF
    rrf_k: int = 60
    primary_results_weight: float = 0.5
    secondary_results_weight: float = 0.5
    primary_top_k: int = 4
    secondary_top_k: int = 4
    scorer: str = "BM25"
    tsquery_function: str = "plainto_tsquery"
