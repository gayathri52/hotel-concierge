# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 3 · Develop, trace and register the agent
# MAGIC `concierge/agent.py` is an MLflow **ResponsesAgent** (models-from-code). The LLM decides which tools to call:
# MAGIC `search_hotel_knowledge` (Vector Search), `check_dining_availability` / `check_spa_availability` / `lookup_reservation` (SQL),
# MAGIC `get_weather_forecast` (Open-Meteo API).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(".."))
import mlflow
from concierge.agent import AGENT
from concierge.config import load

cfg = load()
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

EXAMPLES = [
    ("MTL", "What restaurants and amenities are available at the hotel?"),
    ("WHI", "Does the hotel have a spa or family activities?"),
    ("TOR", "What time does breakfast start?"),
    ("WHI", "What should I do tomorrow if it rains?"),
    ("MTL", "I am arriving at 8 PM. What dining options are still available?"),
]
for hotel, q in EXAMPLES:
    r = AGENT.predict({"input": [{"role": "user", "content": q}], "custom_inputs": {"hotel_id": hotel}})
    print(f"### [{hotel}] {q}\n{r.output[0].content[0]['text']}\n"
          f"tools: {[t['tool'] for t in r.custom_outputs['tools_used']]}\n"
          f"sources: {[s.get('doc_id') or s.get('source') for s in r.custom_outputs['sources']]}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log + register to Unity Catalog
# MAGIC `resources` enables **automatic auth passthrough**: the serving endpoint gets short-lived, least-privilege
# MAGIC credentials for exactly these resources — no PATs in the model.

# COMMAND ----------


from mlflow.models.resources import (DatabricksServingEndpoint, DatabricksSQLWarehouse, DatabricksTable,
                                     DatabricksVectorSearchIndex)

resources = [
    DatabricksServingEndpoint(endpoint_name=cfg["llm_endpoint"]),
    DatabricksServingEndpoint(endpoint_name=cfg["embedding_endpoint"]),
    DatabricksVectorSearchIndex(index_name=cfg["vs_index"]),
    DatabricksSQLWarehouse(warehouse_id=cfg["warehouse_id"]),
    *[DatabricksTable(table_name=cfg[f"{t}_table"]) for t in ("restaurants", "spa_slots", "reservations")],
]
model_cfg = {k: v for k, v in cfg.items() if not k.endswith("_table") and k not in ("vs_index", "uc_model")}

with mlflow.start_run(run_name="four-seasons-agent"):
    info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=os.path.abspath("../concierge/agent.py"),
        code_paths=[os.path.abspath("../concierge")],
        model_config=model_cfg,
        resources=resources,
        input_example={"input": [{"role": "user", "content": "What time does breakfast start?"}],
                       "custom_inputs": {"hotel_id": "MTL"}},
        pip_requirements=["mlflow>=3.4", "databricks-sdk>=0.60", "openai>=1.40", "requests", "pyyaml", "httpx"],
        registered_model_name=cfg["uc_model"],
    )
print(cfg["uc_model"], "version", info.registered_model_version)

# COMMAND ----------

# Validate the packaged model in an isolated env before serving
mlflow.models.predict(model_uri=info.model_uri, input_data={"input": [{"role": "user", "content": "Is there a spa?"}],
                      "custom_inputs": {"hotel_id": "TOR"}}, env_manager="uv")

# COMMAND ----------

