from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# =============================
# Env
# =============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"
ORDERS_CSV = DATA_DIR / "orders.csv"


# =============================
# Helpers
# =============================
def normalize_order_id(raw: str) -> str:
    # normalize: trim, remove backticks/quotes, upper-case, remove extra spaces
    s = (raw or "").strip()
    s = s.strip("`'\"")
    s = re.sub(r"\s+", "", s)
    return s.upper()


def mask_email(email: str) -> str:
    # mask like l***@domain
    email = (email or "").strip()
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if not local:
        return f"***@{domain}"
    return f"{local[0]}***@{domain}"


def parse_money(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def in_period(ts: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    # inclusive range
    return (ts >= start) and (ts <= end)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CUSTOMERS_CSV.exists():
        raise FileNotFoundError(f"Missing {CUSTOMERS_CSV}")
    if not ORDERS_CSV.exists():
        raise FileNotFoundError(f"Missing {ORDERS_CSV}")

    customers = pd.read_csv(CUSTOMERS_CSV)
    orders = pd.read_csv(ORDERS_CSV)

    # normalize ids
    customers["customer_id"] = customers["customer_id"].astype(str).str.strip()
    orders["order_id"] = orders["order_id"].astype(str).apply(normalize_order_id)
    orders["customer_id"] = orders["customer_id"].astype(str).str.strip()

    # parse numbers / timestamps
    orders["total"] = orders["total"].apply(parse_money)
    orders["created_at"] = pd.to_datetime(orders["created_at"], errors="coerce")

    customers["email"] = customers["email"].astype(str).str.strip()
    orders["status"] = orders["status"].astype(str).str.strip()

    return customers, orders


# =============================
# Tools (business logic)
# =============================
def get_order(order_id: str) -> Dict[str, Any]:
    customers, orders = load_data()
    oid = normalize_order_id(order_id)

    row = orders.loc[orders["order_id"] == oid]
    if row.empty:
        return {"error": f"Order {oid} not found"}

    r = row.iloc[0]
    cid = str(r["customer_id"]).strip()
    cust = customers.loc[customers["customer_id"] == cid]
    email = cust.iloc[0]["email"] if not cust.empty else ""

    return {
        "order_id": oid,
        "status": str(r["status"]),
        "total": float(r["total"]),
        "masked_email": mask_email(email),
    }


def refund_order(order_id: str, amount: float) -> Dict[str, Any]:
    info = get_order(order_id)
    if "error" in info:
        return {"ok": False, "reason": info["error"]}

    status = (info.get("status") or "").lower()
    total = float(info.get("total") or 0.0)
    amt = parse_money(amount)

    if status not in {"settled", "prepping"}:
        return {"ok": False, "reason": f"Refunds only allowed for settled or prepping orders (status={info.get('status')})."}

    if amt <= 0:
        return {"ok": False, "reason": "Amount must be > 0."}

    if amt > total:
        return {"ok": False, "reason": f"Amount exceeds order total (amount={amt:.2f}, total={total:.2f})."}

    # In a real system we'd write to DB; here we just return ok.
    return {"ok": True}


def spend_in_period(customer_id: str, start: str, end: str) -> Dict[str, Any]:
    customers, orders = load_data()
    cid = (customer_id or "").strip()

    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return {"error": "Invalid start/end date. Use YYYY-MM-DD."}

    o = orders.loc[orders["customer_id"] == cid].copy()
    o = o.loc[o["created_at"].notna()]

    total_spend = 0.0
    for _, r in o.iterrows():
        if in_period(r["created_at"], start_ts, end_ts):
            total_spend += float(r["total"])

    return {"customer_id": cid, "total_spend": round(total_spend, 2)}


# =============================
# OpenAI function-calling loop
# =============================
TOOLS = [
    {
        "type": "function",
        "name": "get_order",
        "description": "Lookup an order by order_id and return status, total, and masked email.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "type": "function",
        "name": "refund_order",
        "description": "Refund credits for an order, respecting refund rules (status must be settled/prepping, amount <= total).",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["order_id", "amount"],
        },
    },
    {
        "type": "function",
        "name": "spend_in_period",
        "description": "Compute total spend for a customer in an inclusive date period (YYYY-MM-DD).",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
            },
            "required": ["customer_id", "start", "end"],
        },
    },
]


def dispatch_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "get_order":
        return get_order(args.get("order_id", ""))
    if name == "refund_order":
        return refund_order(args.get("order_id", ""), args.get("amount", 0))
    if name == "spend_in_period":
        return spend_in_period(args.get("customer_id", ""), args.get("start", ""), args.get("end", ""))
    return {"error": f"Unknown tool: {name}"}


INJECTION_BLOCK = [
    r"secrets\/",
    r"\bsecrets?\b",
    r"\breveal\b",
    r"\bdump\b",
    r"\bprint\b",
]
REFUSAL_ONE_SENTENCE = "I can’t help with requests to reveal secret files or contents."


def looks_like_injection(q: str) -> bool:
    q = (q or "").lower()
    return any(re.search(p, q) for p in INJECTION_BLOCK)


def run_agent(question: str) -> Dict[str, Any]:
    if looks_like_injection(question):
        # Acceptance: one sentence, no tool calls
        return {"final_answer": REFUSAL_ONE_SENTENCE, "tool_calls": []}

    client = OpenAI(api_key=OPENAI_API_KEY)

    # tool-calling flow per official guide: call -> execute -> append function_call_output -> call again :contentReference[oaicite:1]{index=1}
    input_list: List[Dict[str, Any]] = [{"role": "user", "content": question}]

    resp1 = client.responses.create(
        model=OPENAI_MODEL,
        tools=TOOLS,
        input=input_list,
    )

    input_list += resp1.output
    tool_calls: List[Dict[str, Any]] = []

    for item in resp1.output:
        if getattr(item, "type", None) == "function_call":
            name = item.name
            try:
                args = json.loads(item.arguments or "{}")
            except Exception:
                args = {}

            result = dispatch_tool(name, args)

            tool_calls.append({"tool": name, "args": args, "result": result})

            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result),
                }
            )

    resp2 = client.responses.create(
        model=OPENAI_MODEL,
        tools=TOOLS,
        input=input_list,
        instructions=(
            "Answer the user. Use tool outputs. "
            "If an order email is shown, it must be masked (already masked by tools)."
        ),
    )

    final_answer = (resp2.output_text or "").strip()
    return {"final_answer": final_answer, "tool_calls": tool_calls}


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Starship Coffee - Task 2", layout="wide")
st.title("Starship Coffee — Task 2: Function Calling over Customer Data")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY is missing in .env. Add it, then rerun Streamlit.")

left, right = st.columns(2)

with left:
    st.subheader("Question")
    if "q" not in st.session_state:
        st.session_state.q = ""

    preset1 = "Total spend for customer C-101 from 2025-09-01 to 2025-09-30."
    preset2 = "Refund 5.40 credits for order B77."
    preset3 = "What is the status and masked email for order `  c9  `?"

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Preset 1"):
            st.session_state.q = preset1
    with c2:
        if st.button("Preset 2"):
            st.session_state.q = preset2
    with c3:
        if st.button("Preset 3"):
            st.session_state.q = preset3

    st.session_state.q = st.text_area("Ask about customers/orders", value=st.session_state.q, height=140)
    go = st.button("Run")

with right:
    st.subheader("Output")
    if go:
        if not OPENAI_API_KEY:
            st.error("Missing OPENAI_API_KEY")
            st.stop()

        with st.status("Running tool-calling…", expanded=False):
            out = run_agent(st.session_state.q)

        st.write(out["final_answer"])

        st.subheader("Tool calls")
        if out["tool_calls"]:
            st.dataframe(out["tool_calls"], use_container_width=True)
        else:
            st.caption("No tool calls.")

        st.subheader("JSON")
        st.json(out)
