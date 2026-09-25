# Four Seasons Concierge — AI hotel concierge agent on Databricks

A working prototype of an AI concierge branded as **Four Seasons Hotels and Resorts**, covering three properties: **Four Seasons Hotel Montréal (MTL)**, **Four Seasons Hotel Toronto (TOR)** and **Four Seasons Resort Whistler (WHI)**.

> **Demo data only.** The knowledge base is synthetic. Venues, hours, prices, policies and guests are invented and do not reflect the real Four Seasons properties. This project is not affiliated with or endorsed by Four Seasons.

It is built on Databricks with the Databricks SDK:

| Requirement | Implementation |
|---|---|
| RAG and vector DB | Mosaic AI **Vector Search**. A Delta Sync index uses managed `databricks-gte-large-en` embeddings, **hybrid** (vector + keyword) retrieval and a `hotel_id` metadata filter |
| Agent and tools | MLflow **ResponsesAgent**. The LLM runs a tool-calling loop on a Foundation Model API. It has 5 tools: KB search (Vector Search), dining availability, spa availability and reservation lookup (SQL via the Statement Execution API), and weather (Open-Meteo REST API) |
| API | **Model Serving** endpoint with AI Gateway , plus an optional **Databricks App**  that returns `{answer, tools_used, sources, latency_ms}` |
| Evaluation | `mlflow.genai.evaluate` on 18 labelled cases. It uses 5 rubric-based LLM judges, a deterministic tool-selection scorer and the built-in Safety and Guidelines judges |


## Layout
```
concierge/
  config.yaml      workspace settings (catalog, warehouse, endpoints)
  data.py          synthetic Knowledge Base: 3 hotels, restaurants, spa, policies, activities, FAQs, reservations
  tools.py         tool specs + implementations (Vector Search, SQL, weather API)
  agent.py         ResponsesAgent: system prompt, tool loop, sources/tools_used output
  evaluation.py    eval dataset + LLM-judge scorers
notebooks/         01 build KB -> 02 vector index -> 03 agent dev/log -> 04 evaluate -> 05 deploy/query
app/               FastAPI Databricks App 
```

## API
**Serving endpoint:** `POST https://<workspace>/serving-endpoints/four-seasons-agent/invocations`
```json
{"input": [{"role": "user", "content": "I am arriving at 8 PM. What dining options are still available?"}],
 "custom_inputs": {"hotel_id": "MTL"}}
```
The response is shortened here, and the answer text is illustrative (tools and sources are the real shapes):
```json
{"output": [{"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Welcome! After 8 PM you can still enjoy: ..."}]}],
 "custom_outputs": {
   "tools_used": [{"tool": "check_dining_availability", "arguments": {"time": "20:00", "date": "2026-09-22"}, "status": "ok"}],
   "sources": [{"tool": "check_dining_availability", "source": "four_seasons.hotel_concierge.restaurants"},
               {"doc_id": "MTL-dining-027", "hotel_id": "MTL", "title": "Restaurant: Verrière", "score": 0.83}]}}
```

## Agent design
- **Tool routing is done by the LLM.** It reads the tool descriptions and a system prompt with routing rules. For example: "what's open after X" → SQL dining tool, a forecast → weather API, any hotel fact → KB search. The loop is capped at `max_steps` iterations.
- **The RAG and SQL data can't drift apart.** Restaurant and spa KB documents are generated from the same records as the SQL tables. Each chunk embeds `hotel name | section title` so it describes itself.
- **Grounding:** the prompt requires facts to come from tools. The agent says "I don't have that information" rather than guessing. `sources` lists every retrieved document and table it used.
- **Security in code:** the agent drops client-supplied `system` messages, strips arguments that aren't in the tool schema, and uses parameterised SQL (no string interpolation of user input). Reservation lookup requires the confirmation number *and* last name. Email and phone are never selected, so they can't reach the LLM. Tool errors go back to the LLM instead of crashing the request.

## Evaluation approach
The notebook calls `mlflow.genai.evaluate(data=eval_set(), predict_fn=..., scorers=SCORERS)`. There are 18 cases: the 5 required questions across properties, plus late arrival, spa slots, weather, transport, pets, a verified reservation, **PII probing**, **prompt injection**, an **out-of-KB** question (helipad) and a cross-property question. Each case has hand-written `expected_facts`, `required_tools` and `optional_tools`.

| Metric | Scale | What the judge sees | Why |
|---|---|---|---|
| groundedness | 1–5 | answer + **tool outputs read from the MLflow trace** | Is every claim supported by what was actually retrieved? |
| hallucination_free | bool | answer + tool outputs | Hard fail on any invented hotel fact |
| relevance | 1–5 | question, answer | Does it answer what was asked? |
| overall_quality | 1–5 | question, answer | Helpful, warm, concise, correct time format |
| tool_selection | bool, no LLM | tools used vs. expected | `required / optional` |
| safety, policy_compliance | bool | Databricks built-in judges | Harmful content; no prompt leak or PII disclosure |

**Scoring choices:**
- **One criterion per judge.** Each judge has an explicit anchored rubric, which is more reliable than asking for one "overall" score.
- **The judge model family differs from the agent's** (`judge_endpoint`), which reduces self-preference bias.
- **Tool selection is deterministic**, since it has an objective answer.
- **Suggested release gate:** mean correctness ≥ 4.0, hallucination_free ≥ 95%, tool_selection ≥ 90%, safety and policy at 100%. Calibrate the judges by hand-labelling around 30 traces and checking agreement.

The notebooks themselves need a workspace to run.
