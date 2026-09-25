import os
import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
TABLES = ["hotels", "kb_docs", "restaurants", "spa_slots", "reservations"]


def resolve(cfg: dict) -> dict:
    """Derive fully-qualified UC names from catalog/schema."""
    c = dict(cfg)
    fq = f"{c['catalog']}.{c['schema']}"
    for t in TABLES:
        c.setdefault(f"{t}_table", f"{fq}.{t}")
    c.setdefault("vs_index", f"{fq}.kb_docs_index")
    c.setdefault("uc_model", f"{fq}.concierge_agent")
    return c


def load(path: str = DEFAULT_PATH) -> dict:
    with open(path) as f:
        return resolve(yaml.safe_load(f))
