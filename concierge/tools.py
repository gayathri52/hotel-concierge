"""Concierge tools. Every tool is traced as a TOOL span (used later by the groundedness judge)."""
import hashlib
import json
from datetime import date as _date, datetime

import mlflow
import requests
from databricks.sdk.service.sql import StatementParameterListItem, StatementState
from mlflow.entities import SpanType

from concierge.data import HOTELS

_HOTEL = {"type": "string", "enum": list(HOTELS), "description": "Hotel id. Defaults to the guest's current hotel."}
_DATE = {"type": "string", "description": "Date as YYYY-MM-DD (hotel local)."}


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


TOOL_SPECS = [
    _fn("search_hotel_knowledge",
        "Semantic + keyword search over the hotel knowledge base: restaurants, menus/hours, amenities, spa, "
        "family programs, policies, rainy-day and local activities, transportation, FAQs. Use for any hotel fact.",
        {"query": {"type": "string"}, "hotel_id": _HOTEL,
         "num_results": {"type": "integer", "description": "1-5, default 5"}}, ["query"]),
    _fn("check_dining_availability",
        "SQL lookup of dining venues open at or after a given local time (e.g. a late arrival).",
        {"hotel_id": _HOTEL, "time": {"type": "string", "description": "24h HH:MM"}, "date": _DATE}, ["time"]),
    _fn("check_spa_availability",
        "SQL lookup of bookable spa treatment slots for a date.",
        {"hotel_id": _HOTEL, "date": _DATE, "treatment": {"type": "string", "description": "Optional name filter, e.g. 'massage'"}}, ["date"]),
    _fn("get_weather_forecast",
        "Daily weather forecast (up to 14 days ahead) at the hotel's location.",
        {"hotel_id": _HOTEL, "date": _DATE}, []),
    _fn("lookup_reservation",
        "Look up the guest's own reservation. Requires BOTH confirmation number and last name as given by the guest.",
        {"confirmation_number": {"type": "string"}, "last_name": {"type": "string"}}, ["confirmation_number", "last_name"]),
]
TOOL_PARAMS = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in TOOL_SPECS}

WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers",
       82: "heavy rain showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def _hhmm(t: str) -> str:
    t = t.strip().upper().replace(".", "")
    for f in ("%H:%M", "%I %p", "%I:%M %p", "%I%p", "%H"):
        try:
            return datetime.strptime(t, f).strftime("%H:%M")
        except ValueError:
            pass
    raise ValueError(f"Unrecognised time '{t}', use HH:MM")


class ConciergeTools:
    def __init__(self, cfg: dict, w):
        self.cfg, self.w = cfg, w

    def call(self, name: str, args: dict, ctx: dict) -> dict:
        if name not in TOOL_PARAMS:
            return {"error": f"Unknown tool '{name}'"}
        args = {k: v for k, v in args.items() if k in TOOL_PARAMS[name]}  # drop hallucinated args
        if "hotel_id" in TOOL_PARAMS[name]:
            args["hotel_id"] = (args.get("hotel_id") or ctx.get("hotel_id") or "").upper() or None
            if name != "search_hotel_knowledge" and args["hotel_id"] not in HOTELS:
                return {"error": "hotel_id required: one of " + ", ".join(HOTELS)}
        try:
            return getattr(self, name)(**args)
        except Exception as e:  # tool failures are returned to the LLM, never raised to the guest
            return {"error": f"{type(e).__name__}: {e}"}

    # ---------- RAG ----------
    @mlflow.trace(span_type=SpanType.TOOL)
    def search_hotel_knowledge(self, query: str, hotel_id: str | None = None, num_results: int = 5):
        filters = {"hotel_id": [hotel_id, "ALL"]} if hotel_id else None
        r = self.w.vector_search_indexes.query_index(
            index_name=self.cfg["vs_index"],
            columns=["doc_id", "hotel_id", "category", "title", "content"],
            query_text=query, query_type="HYBRID",
            num_results=max(1, min(int(num_results or 5), 8)),
            filters_json=json.dumps(filters) if filters else None)
        cols = [c.name for c in r.manifest.columns]
        hits = [dict(zip(cols, row)) for row in (r.result.data_array or [])]
        for h in hits:
            h["score"] = round(float(h.get("score", 0)), 4)
        return {"results": hits}

    # ---------- SQL ----------
    def _sql(self, stmt: str, **params) -> list[dict]:
        r = self.w.statement_execution.execute_statement(
            statement=stmt, warehouse_id=self.cfg["warehouse_id"], wait_timeout="30s",
            parameters=[StatementParameterListItem(name=k, value=str(v)) for k, v in params.items()])
        if r.status.state != StatementState.SUCCEEDED:
            raise RuntimeError(r.status.error.message if r.status.error else str(r.status.state))
        cols = [c.name for c in r.manifest.schema.columns]
        return [dict(zip(cols, row)) for row in ((r.result and r.result.data_array) or [])]

    @mlflow.trace(span_type=SpanType.TOOL)
    def check_dining_availability(self, hotel_id: str, time: str, date: str | None = None):
        t = _hhmm(time)
        rows = self._sql(
            f"SELECT venue, cuisine, meal_period, days, open_time, last_seating, reservation_required, dress_code "
            f"FROM {self.cfg['restaurants_table']} WHERE hotel_id = :h AND last_seating >= :t ORDER BY open_time",
            h=hotel_id, t=t)
        if date:  # drop services that don't run on that weekday
            wd = _date.fromisoformat(date).strftime("%a")
            rows = [r for r in rows if _runs_on(r["days"], wd)]
        for r in rows:
            r["status"] = "open at that time" if r["open_time"] <= t else f"opens at {r['open_time']}"
        return {"hotel_id": hotel_id, "time": t, "date": date, "venues": rows,
                "source": self.cfg["restaurants_table"]}

    @mlflow.trace(span_type=SpanType.TOOL)
    def check_spa_availability(self, hotel_id: str, date: str, treatment: str | None = None):
        stmt = (f"SELECT treatment, slot_time, duration_min, price, currency FROM {self.cfg['spa_slots_table']} "
                f"WHERE hotel_id = :h AND slot_date = CAST(:d AS DATE) AND is_available")
        params = dict(h=hotel_id, d=date)
        if treatment:
            stmt += " AND lower(treatment) LIKE lower(:tr)"
            params["tr"] = f"%{treatment}%"
        rows = self._sql(stmt + " ORDER BY slot_time, treatment", **params)
        return {"hotel_id": hotel_id, "date": date, "available_slots": rows, "source": self.cfg["spa_slots_table"]}

    @mlflow.trace(span_type=SpanType.TOOL)
    def lookup_reservation(self, confirmation_number: str, last_name: str):
        # data minimisation: email/phone are never selected, so they can never reach the LLM
        rows = self._sql(
            f"SELECT confirmation_number, first_name, hotel_id, arrival_date, departure_date, room_type, adults, "
            f"children, loyalty_tier, preferences FROM {self.cfg['reservations_table']} "
            f"WHERE confirmation_number = :c AND lower(last_name) = lower(:l)",
            c=confirmation_number.strip().upper(), l=last_name.strip())
        if not rows:
            return {"found": False, "message": "No reservation matches that confirmation number and last name."}
        return {"found": True, "reservation": rows[0], "source": self.cfg["reservations_table"]}

    # ---------- External API ----------
    @mlflow.trace(span_type=SpanType.TOOL)
    def get_weather_forecast(self, hotel_id: str, date: str | None = None):
        h = HOTELS[hotel_id]
        d = date or _date.today().isoformat()
        if self.cfg.get("weather_mode") == "mock":
            n = int(hashlib.md5(f"{hotel_id}{d}".encode()).hexdigest(), 16)
            code, prob, tmax = [0, 2, 3, 61, 63, 80, 95][n % 7], n % 100, 5 + n % 25
            daily = {"weather_code": [code], "temperature_2m_max": [tmax], "temperature_2m_min": [tmax - 8],
                     "precipitation_probability_max": [prob], "precipitation_sum": [round((n % 20) / 2, 1)]}
            src = "mock"
        else:
            r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=6, params=dict(
                latitude=h["lat"], longitude=h["lon"], timezone=h["tz"], start_date=d, end_date=d,
                daily="weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum"))
            r.raise_for_status()
            daily, src = r.json()["daily"], "open-meteo.com"
        code, prob = daily["weather_code"][0], daily["precipitation_probability_max"][0] or 0
        return {"hotel_id": hotel_id, "date": d, "conditions": WMO.get(code, f"code {code}"),
                "temp_max_c": daily["temperature_2m_max"][0], "temp_min_c": daily["temperature_2m_min"][0],
                "precipitation_probability_pct": prob, "precipitation_mm": daily["precipitation_sum"][0],
                "rain_likely": prob >= 50 or code in (61, 63, 65, 80, 81, 82, 95, 96, 99), "source": src}


def _runs_on(days: str, wd: str) -> bool:
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if days == "Daily":
        return True
    a, b = days.split("-")
    i, j, k = order.index(a), order.index(b), order.index(wd)
    return i <= k <= j if i <= j else k >= i or k <= j
