# langchain-serenedb

A [LangChain](https://python.langchain.com/) vector store integration for
[SereneDB](https://serenedb.com)

SereneDB speaks the PostgreSQL wire protocol, so this package connects with **psycopg3**.
It maps the integration onto SereneDB's native capabilities:

| Vector Store Search Feature | SereneDB Feature used |
|---|---|
| Vector column |  `FLOAT[N]`  |
| Distance ops | `<->`, `<=>`, `<#>`, `<+>` |
| ANN index |  inverted index on the vector column e.g. `USING inverted (emb hnsw (metric='cosine', ...))` |
| Full-text |  inverted index on the text column + `BM25(idx.tableoid)` |
| Metadata | `JSON` column, explicit columns |

## Installation

```bash
pip install langchain-serenedb
```

Requires Python 3.10+. Not yet published to PyPI — until the first release, install from
a checkout (`pip install -e .`; see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Quickstart (engine + table)

```python
from langchain_serenedb import SereneDBEngine, HNSWIndex

engine = SereneDBEngine.from_connection_string(
    "host=127.0.0.1 port=7890 user=postgres dbname=postgres"
)

# Table only (vector search falls back to an exact scan until an index is built):
engine.init_vectorstore_table(table_name="my_docs", vector_size=768)

# Or create the table and its HNSW ANN index in one call, so vector search is
# accelerated from the start:
engine.init_vectorstore_table("my_docs", 768, vector_index=HNSWIndex())
#   ...or the combined full-text + vector index for hybrid search:
#   engine.init_vectorstore_table("my_docs", 768, hybrid_search_config=HybridSearchConfig())

# ... after writing rows, publish them to the inverted index:
engine.refresh_table("my_docs")
```

> **Tip:** for a large bulk load, SereneDB builds a more compact graph if you create the
> index *after* loading (`store.apply_vector_index(HNSWIndex())`); creating it up front
> with the table is the convenient choice for incremental workloads.

## Contributing

Building, testing, linting, and running the suite (locally or in Docker Compose) are
covered in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
