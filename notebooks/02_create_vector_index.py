# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 2 · Vector Search index (Databricks SDK)
# MAGIC Delta Sync index with **managed embeddings** (`databricks-gte-large-en`) over `kb_docs.content`.
# MAGIC Updating the KB = update the Delta table + `sync_index` (no re-embedding code to maintain).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import json, os, sys, time
sys.path.append(os.path.abspath(".."))
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.vectorsearch import (DeltaSyncVectorIndexSpecRequest, EmbeddingSourceColumn,
                                                 EndpointType, PipelineType, VectorIndexType)
from concierge.config import load

cfg, w = load(), WorkspaceClient()

# COMMAND ----------

if cfg["vs_endpoint"] not in {e.name for e in w.vector_search_endpoints.list_endpoints()}:
    w.vector_search_endpoints.create_endpoint_and_wait(name=cfg["vs_endpoint"], endpoint_type=EndpointType.STANDARD)

try:
    w.vector_search_indexes.get_index(cfg["vs_index"])
    w.vector_search_indexes.sync_index(cfg["vs_index"])
    print("Index exists -> sync triggered")
except NotFound:
    w.vector_search_indexes.create_index(
        name=cfg["vs_index"], endpoint_name=cfg["vs_endpoint"], primary_key="doc_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=cfg["kb_docs_table"], pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[EmbeddingSourceColumn(name="content", embedding_model_endpoint_name=cfg["embedding_endpoint"])],
            columns_to_sync=["doc_id", "hotel_id", "category", "title", "content"]))
    print("Index created")

while not (idx := w.vector_search_indexes.get_index(cfg["vs_index"])).status.ready:
    print("waiting:", idx.status.message); time.sleep(30)
print("ready:", idx.status.indexed_row_count, "rows")

# COMMAND ----------

# Retrieval smoke test: hybrid (vector + keyword) search with a metadata filter
r = w.vector_search_indexes.query_index(
    index_name=cfg["vs_index"], columns=["doc_id", "title", "content"], query_text="what time is breakfast",
    query_type="HYBRID", num_results=3, filters_json=json.dumps({"hotel_id": ["MTL", "ALL"]}))
for row in r.result.data_array:
    print(row[0], "|", row[1], "| score", row[-1])

# COMMAND ----------

