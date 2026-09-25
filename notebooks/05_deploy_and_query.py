# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 5 · Deploy (Databricks SDK) and expose the API
# MAGIC 1. Model Serving endpoint for the registered agent, with AI Gateway inference tables, usage tracking and per-user rate limits.
# MAGIC 2. Databricks App (`app/`) — a clean REST contract: `POST /ask -> {answer, tools_used, sources, latency_ms}`.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os, sys, time
sys.path.append(os.path.abspath(".."))
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (AiGatewayConfig, AiGatewayInferenceTableConfig,
                                            EndpointCoreConfigInput, ServedEntityInput)
from concierge.config import load

cfg, w = load(), WorkspaceClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
_versions = mlflow.MlflowClient().search_model_versions(f"name='{cfg['uc_model']}'")
if not _versions:
    raise RuntimeError(f"No registered model versions found for '{cfg['uc_model']}'. Run the model registration notebook (03 or 04) first.")
version = max(int(v.version) for v in _versions)
exp = mlflow.set_experiment(f"/Users/{w.current_user.me().user_name}/four-seasons-concierge")

entity = ServedEntityInput(
    entity_name=cfg["uc_model"], entity_version=str(version), workload_size="Small", scale_to_zero_enabled=True,
    environment_vars={"ENABLE_MLFLOW_TRACING": "true", "MLFLOW_EXPERIMENT_ID": exp.experiment_id})
gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(catalog_name=cfg["catalog"], schema_name=cfg["schema"],
                                                         table_name_prefix="concierge", enabled=True))

name = cfg["serving_endpoint"]
try:
    w.serving_endpoints.get(name)
    w.serving_endpoints.update_config_and_wait(name, served_entities=[entity])
except NotFound:
    w.serving_endpoints.create_and_wait(name=name, config=EndpointCoreConfigInput(served_entities=[entity]), ai_gateway=gateway)
try:
    w.serving_endpoints.put_ai_gateway(name, inference_table_config=gateway.inference_table_config)
except NotFound:
    print("AI Gateway inference tables not supported on this endpoint; skipping.")
print("endpoint ready:", name, "v", version)

# COMMAND ----------

# Query the endpoint (same REST call any client makes: POST /serving-endpoints/<name>/invocations)
t = time.time()
resp = w.api_client.do("POST", f"/serving-endpoints/{name}/invocations", body={
    "input": [{"role": "user", "content": "I am arriving at 8 PM. What dining options are still available?"}],
    "custom_inputs": {"hotel_id": "MTL"}})
print(resp["output"][0]["content"][0]["text"])
print("tools:", resp["custom_outputs"]["tools_used"])
print("sources:", resp["custom_outputs"]["sources"])
print(f"latency: {time.time() - t:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the REST wrapper as a Databricks App

# COMMAND ----------

from databricks.sdk.service.apps import (App, AppDeployment, AppResource, AppResourceServingEndpoint,
                                         AppResourceServingEndpointServingEndpointPermission)

app_name = "four-seasons-api"
src = os.path.abspath("../app")  # workspace path of the app/ folder
try:
    w.apps.get(app_name)
except NotFound:
    w.apps.create_and_wait(App(name=app_name, description="Four Seasons concierge REST API", resources=[
        AppResource(name="agent-endpoint", serving_endpoint=AppResourceServingEndpoint(
            name=name, permission=AppResourceServingEndpointServingEndpointPermission.CAN_QUERY))]))
d = w.apps.deploy_and_wait(app_name, AppDeployment(source_code_path=src))
print(d.status.message, "->", w.apps.get(app_name).url)

# COMMAND ----------

