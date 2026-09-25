# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 4 · Evaluate with LLM-as-a-Judge
# MAGIC 18 labelled cases (the 5 required questions across properties + late arrival, spa, weather, reservations,
# MAGIC PII probing, prompt injection, out-of-KB, cross-property). Scorers live in `concierge/evaluation.py`.
# MAGIC
# MAGIC | Metric | Type | How it's scored |
# MAGIC |---|---|---|
# MAGIC | retrieval_relevance | LLM judge 0-1 | share of retrieved chunks relevant to the question (precision@k) |
# MAGIC | groundedness | LLM judge 1-5 | answer vs. **tool outputs pulled from the MLflow trace** |
# MAGIC | completeness | LLM judge 1-5 | every part of the question answered, using what the tools returned |
# MAGIC | hallucination_free | LLM judge bool | any hotel fact not present in tool outputs → fail |
# MAGIC | relevance | LLM judge 1-5 | does it answer the question asked |
# MAGIC | overall_quality | LLM judge 1-5 | helpful, tone, concise, actionable |
# MAGIC | tool_selection | deterministic bool | required / optional |
# MAGIC | safety, policy_compliance | Databricks built-in judges | harmful content; PII / prompt-leak guidelines |

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Cell 3
import os, sys
sys.path.append(os.path.abspath(".."))
import mlflow
import pandas as pd
from concierge.agent import AGENT
from concierge.evaluation import SCORERS, eval_set


def predict_fn(question, hotel_id=None, now=None):
    return AGENT.answer([{"role": "user", "content": question}],
                        {k: v for k, v in {"hotel_id": hotel_id, "now": now}.items() if v})


# Throttle parallelism: by default 10 cases x 10 scorer threads hit the FM API at once, which trips
# pay-per-token rate limits (especially on Free Edition). Raise these if your workspace has higher limits.
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "2"
os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "2"

mlflow.set_experiment("/Users/gayathri96.ca@gmail.com/fourseasons-concierge/four-seasons-eval")

data = eval_set()
with mlflow.start_run(run_name="four-seasons-eval"):
    results = mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=SCORERS)

pd.DataFrame([results.metrics]).T.rename(columns={0: "value"})

# COMMAND ----------

# Per-case scores (open the run's "Evaluations" tab for full traces + judge rationales)
df = results.result_df
display(df[["request"] + [c for c in df.columns if c.endswith("/value")]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Judge justifications
# MAGIC Every score comes with the judge's evidence-based rationale (`<metric>/rationale`), shown side by side with the value.

# COMMAND ----------

# metrics = [c[:-len("/value")] for c in df.columns if c.endswith("/value")
#            and c[:-len("/value")] not in ("expected_facts", "required_tools", "optional_tools")]


# def _q(r):
#     return r.get("question") if isinstance(r, dict) else r


# def _a(r):
#     return r.get("answer") if isinstance(r, dict) else r


# report = df.assign(question=df["request"].map(_q), answer=df["response"].map(_a))[
#     ["question", "answer"] + [col for m in metrics for col in (f"{m}/value", f"{m}/rationale") if col in df.columns]]
# display(report)

# COMMAND ----------

# Readable per-question report card
for _, row in report.iterrows():
    print("=" * 100, f"\nQ: {row['question']}\nA: {str(row['answer'])[:400]}\n")
    for m in metrics:
        v, why = row.get(f"{m}/value"), row.get(f"{m}/rationale")
        if v is not None or why:
            print(f"  {m:20s} {str(v):6s} | {why or ''}")

# COMMAND ----------

