from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Env + paths
# -----------------------------
load_dotenv()  # reads .env from project root

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()

HERE = Path(__file__).resolve().parent
RECEIPTS_DIR = HERE / "receipts"
OUT_JSONL = HERE / "predictions.jsonl"
OUT_JSONL.parent.mkdir(exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------
# Helpers
# -----------------------------
def to_data_url(file_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json(text: str) -> Dict[str, Any]:
    """
    Robust-ish JSON extraction: find first {...} block and parse.
    """
    text = text.strip()
    # If model returned pure JSON, parse directly
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Model did not return JSON.")
    return json.loads(m.group(0))


def validate_shape(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "items" not in obj or "total" not in obj:
        raise ValueError("JSON must contain keys: items, total")

    items = obj["items"]
    if not isinstance(items, list):
        raise ValueError("items must be a list")

    clean_items: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        clean_items.append(
            {
                "name": str(it.get("name", "")).strip(),
                "qty": int(it.get("qty", 1) or 1),
                "unit_price": str(it.get("unit_price", "0.00")).strip(),
                "line_total": str(it.get("line_total", "0.00")).strip(),
            }
        )

    return {"items": clean_items, "total": str(obj["total"]).strip()}


def save_jsonl(record: Dict[str, Any]) -> None:
    with OUT_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_vision_receipt(image_bytes: bytes, mime: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    prompt = (
        "You are extracting structured data from a receipt image.\n"
        "Return ONLY valid JSON with this exact shape:\n"
        "{\n"
        '  "items": [ {"name":"...", "qty": 1, "unit_price":"0.00", "line_total":"0.00"} ],\n'
        '  "total": "0.00"\n'
        "}\n"
        "Rules:\n"
        "- Keep numbers as strings with 2 decimals for unit_price/line_total/total.\n"
        "- If multiple totals appear (e.g., crossed-out and current), return the CURRENT total.\n"
        "- If qty is missing, assume 1.\n"
        "- Do not add extra keys.\n"
    )

    data_url = to_data_url(image_bytes, mime)

    resp = client.responses.create(
        model=VISION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    raw = resp.output_text.strip()
    obj = extract_json(raw)
    obj = validate_shape(obj)
    return obj


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Starship Coffee - Task 3 Receipts", layout="wide")
st.title("Starship Coffee — Task 3: Receipt OCR (Vision → JSON)")

uploaded = st.file_uploader("Upload a receipt image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    st.image(uploaded, caption=uploaded.name, use_container_width=True)

    if st.button("Extract"):
        try:
            with st.status("Calling vision model…", expanded=False):
                pred = run_vision_receipt(uploaded.getvalue(), uploaded.type)

            # Save jsonl
            record = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "file": uploaded.name,
                "prediction": pred,
            }
            save_jsonl(record)

            st.subheader("JSON")
            st.json(pred)

            st.subheader("Items")
            st.dataframe(pred["items"], use_container_width=True)

            st.subheader("Total")
            st.write(pred["total"])

            st.caption(f"Saved to: {OUT_JSONL}")

        except Exception as e:
            st.error(str(e))
