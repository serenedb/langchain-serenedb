"""Hybrid search configuration for SereneDBVectorStore.

Hybrid search fuses a dense (vector ANN) ranking and a lexical (BM25 full-text) ranking
in a **single** SereneDB query — a ``WITH fused AS (lexical UNION ALL vector)`` CTE whose
outer query combines the two per the chosen :class:`FusionStrategy`. The lexical branch
indexes the ``content`` column in an inverted index (with a text search dictionary
carrying ``frequency = true``) and scores matches with ``BM25(index.tableoid)``.
See :class:`HybridSearchConfig`.
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
class HybridSearchConfig:
    """SereneDB vector store hybrid-search configuration.

    The lexical branch is served by an inverted index on the ``content`` column.
    Scoring requires the column's text search dictionary to carry ``frequency = true``
    (and ``position = true`` for phrase queries, ``norm = true`` for the language-model
    scorers). Because the ``@@`` predicate resolves only when selecting *from the index
    by name*, the store also stores the id/metadata columns via ``INCLUDE`` so the
    lexical branch can return them.

    Attributes:
        fts_query: The full-text query string. If empty, the search query text is used.
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
        dictionary_name: Text search dictionary used to analyze the content column.
        dictionary_options: Options for ``CREATE TEXT SEARCH DICTIONARY`` — must include
            ``frequency = true`` for scoring.
        scorer: Relevance scorer function name (``BM25``, ``TFIDF``, ``dfi``, ...).
        tsquery_function: Query constructor (``plainto_tsquery``, ``to_tsquery``,
            ``phraseto_tsquery`` or ``websearch_to_tsquery``).
    """

    fts_query: str = ""
    fusion: FusionStrategy = FusionStrategy.RRF
    rrf_k: int = 60
    primary_results_weight: float = 0.5
    secondary_results_weight: float = 0.5
    primary_top_k: int = 4
    secondary_top_k: int = 4
    dictionary_name: str = "langchain_fts_dict"
    dictionary_options: str = (
        "template = 'segmentation', case = 'lower', "
        "frequency = true, position = true, norm = true"
    )
    scorer: str = "BM25"
    tsquery_function: str = "plainto_tsquery"
