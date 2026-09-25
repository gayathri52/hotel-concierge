# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 1 · Build the synthetic knowledge base
# MAGIC Writes 5 Unity Catalog Delta tables: `hotels`, `kb_docs` (RAG source), `restaurants`, `spa_slots`, `reservations` (structured tools).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(".."))
from pyspark.sql import Row
from concierge import data
from concierge.config import load

cfg = load()
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg['catalog']}.{cfg['schema']}")

tables = {
    "hotels": data.hotel_rows(),
    "kb_docs": data.kb_docs(),
    "restaurants": data.RESTAURANTS,
    "spa_slots": data.spa_slots(),
    "reservations": data.reservations(),
}
for name, rows in tables.items():
    (spark.createDataFrame([Row(**r) for r in rows])
        .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg[f"{name}_table"]))
    print(f"{cfg[f'{name}_table']}: {len(rows)} rows")

# Delta Sync vector indexes read the Change Data Feed
spark.sql(f"ALTER TABLE {cfg['kb_docs_table']} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
# PII columns are tagged so governance/masking policies can target them
spark.sql(f"ALTER TABLE {cfg['reservations_table']} ALTER COLUMN email SET TAGS ('pii' = 'email')")
spark.sql(f"ALTER TABLE {cfg['reservations_table']} ALTER COLUMN phone SET TAGS ('pii' = 'phone')")

# COMMAND ----------

display(spark.table(cfg["kb_docs_table"]).select("doc_id", "hotel_id", "category", "content"))

# COMMAND ----------

spark.table('four_seasons.hotel_concierge.hotels').display()
spark.table('four_seasons.hotel_concierge.kb_docs').display()
spark.table('four_seasons.hotel_concierge.restaurants').display()
spark.table('four_seasons.hotel_concierge.spa_slots').display()
spark.table('four_seasons.hotel_concierge.reservations').display()
# (main.hotel_conc
# main.hotel_concierge.restaurants: 17 rows
# main.hotel_concierge.spa_slots: 630 rows
# main.hotel_concierge.reservations: 3 rows)

# COMMAND ----------

