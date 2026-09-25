"""Thin REST facade over the agent serving endpoint (runs as a Databricks App).
"""
import os
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ENDPOINT = os.environ["SERVING_ENDPOINT"]
w = WorkspaceClient()  # app service principal (OAuth), injected by Databricks Apps
app = FastAPI(title="Four Seasons Concierge API")


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    hotel_id: str | None = Field(default=None, pattern="^(MTL|WHI|TOR)$")
    history: list[Turn] = Field(default_factory=list, max_length=20)


HERE = Path(__file__).parent
INDEX = (HERE / "index.html").read_text()
if (HERE / "static").is_dir():  # optional extra assets; the logo is embedded in index.html
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return INDEX


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    body = {"input": [t.model_dump() for t in req.history] + [{"role": "user", "content": req.question}],
            "custom_inputs": {"hotel_id": req.hotel_id} if req.hotel_id else {}}
    t0 = time.time()
    try:
        r = w.api_client.do("POST", f"/serving-endpoints/{ENDPOINT}/invocations", body=body)
    except Exception as e:
        raise HTTPException(502, f"Agent unavailable: {type(e).__name__}: {str(e)[:300]}")
    text = next((c["text"] for o in r.get("output", []) if o.get("type") == "message"
                 for c in o.get("content", []) if c.get("type") == "output_text"), "")
    co = r.get("custom_outputs") or {}
    return {"answer": text, "tools_used": co.get("tools_used", []), "sources": co.get("sources", []),
            "latency_ms": int((time.time() - t0) * 1000)}
