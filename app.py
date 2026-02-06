import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

from media_extract import MediaResult, extract_text
from parser import Message, filter_system_messages, parse_chat


st.set_page_config(page_title="WhatsApp Group Insights", layout="wide")


def _ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _call_ollama(model: str, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"].strip()


TICKER_RE = re.compile(r"\$?[A-Z]{1,5}\b")
STOP_TOKENS = {
    "A", "I", "WE", "US", "USA", "THE", "AND", "OR", "FOR", "TO", "IN",
    "ON", "OF", "IS", "ARE", "AM", "BE", "BUY", "SELL", "HOLD", "LONG",
    "SHORT", "CALL", "PUT", "IPO", "ETFs", "ETF", "CEO", "FED", "SEC",
    "GDP", "EPS", "PM", "AM",
}


def _extract_tickers(text: str) -> List[str]:
    found = []
    for tok in TICKER_RE.findall(text):
        raw = tok.replace("$", "").upper()
        if raw in STOP_TOKENS:
            continue
        if len(raw) == 1:
            continue
        found.append(raw)
    return found


def _heuristic_sentiment(messages: List[Message]) -> Dict[str, Dict[str, int]]:
    scores: Dict[str, Dict[str, int]] = defaultdict(lambda: {"buy": 0, "sell": 0, "hold": 0})
    for msg in messages:
        text = msg.text.lower()
        tickers = _extract_tickers(msg.text)
        if not tickers:
            continue
        if any(k in text for k in ["buy", "long", "accumulate", "adding", "bull"]):
            for t in tickers:
                scores[t]["buy"] += 1
        if any(k in text for k in ["sell", "trim", "exit", "bear", "short"]):
            for t in tickers:
                scores[t]["sell"] += 1
        if any(k in text for k in ["hold", "wait", "watch"]):
            for t in tickers:
                scores[t]["hold"] += 1
    return scores


def _format_messages(messages: List[Message], limit: int = 800) -> str:
    lines = []
    for msg in messages[-limit:]:
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M") if msg.timestamp else ""
        author = msg.author or ""
        line = f"[{ts}] {author}: {msg.text}".strip()
        lines.append(line)
    return "\n".join(lines)


def _heuristic_summary(messages: List[Message], top_n: int = 6) -> str:
    tickers = []
    for m in messages:
        tickers.extend(_extract_tickers(m.text))
    ticker_counts = Counter(tickers).most_common(top_n)

    authors = [m.author for m in messages if m.author]
    top_authors = Counter(authors).most_common(5)

    scores = _heuristic_sentiment(messages)
    stance_lines = []
    for t, s in sorted(scores.items(), key=lambda x: (x[1]["buy"] + x[1]["sell"] + x[1]["hold"]), reverse=True)[:top_n]:
        stance = []
        if s["buy"]:
            stance.append(f"buy {s['buy']}")
        if s["sell"]:
            stance.append(f"sell {s['sell']}")
        if s["hold"]:
            stance.append(f"watch {s['hold']}")
        if stance:
            stance_lines.append(f"{t}: " + ", ".join(stance))

    example_lines = []
    for m in reversed(messages):
        if len(example_lines) >= 5:
            break
        if _extract_tickers(m.text):
            ts = m.timestamp.strftime("%Y-%m-%d %H:%M") if m.timestamp else ""
            author = m.author or ""
            snippet = m.text.replace("\n", " ")
            snippet = snippet[:180] + ("…" if len(snippet) > 180 else "")
            example_lines.append(f"[{ts}] {author}: {snippet}".strip())

    parts = []
    if top_authors:
        parts.append("Frequent participants: " + ", ".join(f"{a} ({c})" for a, c in top_authors))
    if ticker_counts:
        parts.append("Most mentioned tickers: " + ", ".join(f"{t} ({c})" for t, c in ticker_counts))
    if stance_lines:
        parts.append("Buy/Sell signals (heuristic): " + " | ".join(stance_lines))
    if example_lines:
        parts.append("Recent ticker messages:\n- " + "\n- ".join(example_lines))
    return "\n\n".join(parts) if parts else "No obvious stock-related signals detected in the selected range."

def _build_media_index(root_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f not in index:
                index[f] = os.path.join(root, f)
    return index


@st.cache_data(show_spinner=False)
def _extract_media_cached(path: str, whisper_model: str) -> MediaResult:
    return extract_text(path, whisper_model=whisper_model or None)


def _messages_with_media_text(
    messages: List[Message],
    media_index: Dict[str, str],
    include_media: bool,
    whisper_model: str,
    max_media: int,
    max_chars: int,
) -> Tuple[List[Message], List[MediaResult]]:
    results: List[MediaResult] = []
    if not include_media:
        return messages, results

    processed = 0
    for msg in messages:
        if not msg.attachments:
            continue
        for name in msg.attachments:
            if processed >= max_media:
                break
            path = media_index.get(name)
            if not path:
                results.append(MediaResult(path=name, status="missing_file", text=""))
                processed += 1
                continue
            res = _extract_media_cached(path, whisper_model)
            results.append(res)
            if res.text:
                snippet = res.text[:max_chars].strip()
                if snippet:
                    msg.text = (msg.text + f"\n[Media {name}] {snippet}").strip()
            processed += 1
        if processed >= max_media:
            break

    return messages, results


def _ollama_insights(messages: List[Message], model: str) -> Tuple[str, List[Dict[str, str]]]:
    system_prompt = (
        "You are a careful financial chat analyst. Extract discussed stocks and summarize"
        " why participants mention buying or selling. Use only the chat content."
        " Provide concise, factual summaries without giving financial advice."
    )
    chat_text = _format_messages(messages)
    user_prompt = (
        "Summarize the group discussion and extract a list of stocks mentioned. "
        "For each stock, provide a short 'why' explanation based on the messages. "
        "Return JSON with keys: summary (string), stocks (array of {ticker, stance, why})."
        " Stance should be one of: buy, sell, watch, mixed, unclear.\n\n"
        f"CHAT:\n{chat_text}"
    )
    raw = _call_ollama(model, system_prompt, user_prompt)
    try:
        data = json.loads(raw)
        summary = data.get("summary", "")
        stocks = data.get("stocks", [])
        clean = []
        for item in stocks:
            if not isinstance(item, dict):
                continue
            clean.append({
                "ticker": str(item.get("ticker", "")).upper(),
                "stance": str(item.get("stance", "")),
                "why": str(item.get("why", "")),
            })
        return summary, clean
    except json.JSONDecodeError:
        # Fallback to raw text if the model doesn't return JSON
        return raw, []


st.title("WhatsApp Group Insights")

st.markdown(
    "This app analyzes an exported WhatsApp group chat file (.txt) and extracts a summary and "
    "stock-related discussion points. It runs locally. For better summaries, install Ollama and "
    "download a model (e.g., `ollama pull llama3.1`)."
)

with st.sidebar:
    st.header("Settings")
    use_ollama = st.checkbox("Use Ollama (local LLM)", value=True)
    model_name = st.text_input("Ollama model", value="llama3.1")
    max_messages = st.slider("Messages to analyze", min_value=200, max_value=5000, value=1200, step=200)
    include_media = st.checkbox("Include media OCR/transcripts", value=True)
    max_media = st.slider("Max media files to process", min_value=10, max_value=500, value=80, step=10)
    max_media_chars = st.slider("Max chars per media item", min_value=200, max_value=4000, value=1200, step=200)
    whisper_model = st.selectbox("Audio model (whisper)", ["", "tiny", "base", "small", "medium"], index=1)

uploaded = st.file_uploader("Upload WhatsApp chat export (.txt) or zip (with media)", type=["txt", "zip"])

if uploaded is None:
    st.info("Upload a WhatsApp exported chat file to begin.")
    st.stop()

chat_text = None
media_root = None

if uploaded.name.lower().endswith(".zip"):
    tmpdir = tempfile.mkdtemp(prefix="whatsapp_export_")
    with zipfile.ZipFile(uploaded) as z:
        z.extractall(tmpdir)
    # Find the chat file
    chat_file = None
    for root, _, files in os.walk(tmpdir):
        for f in files:
            if f.lower() == "_chat.txt" or f.lower().endswith("chat.txt"):
                chat_file = os.path.join(root, f)
                break
        if chat_file:
            break
    if chat_file is None:
        st.error("Could not find _chat.txt inside the zip export.")
        st.stop()
    chat_text = Path(chat_file).read_text(errors="replace")
    media_root = tmpdir
else:
    chat_text = uploaded.read().decode("utf-8", errors="replace")
    media_root = None

lines = chat_text.splitlines()
messages = filter_system_messages(parse_chat(lines))

if not messages:
    st.error("No messages parsed. Make sure this is a WhatsApp exported chat text file.")
    st.stop()

timestamped = [m for m in messages if m.timestamp]
if timestamped:
    min_date = min(m.timestamp for m in timestamped).date()
    max_date = max(m.timestamp for m in timestamped).date()
    preset = st.sidebar.selectbox(
        "Date preset",
        ["Custom", "Last 7 days", "Last 30 days", "Year to date", "All time"],
        index=0,
    )
    if preset == "Last 7 days":
        start_date = max_date - pd.Timedelta(days=6)
        end_date = max_date
    elif preset == "Last 30 days":
        start_date = max_date - pd.Timedelta(days=29)
        end_date = max_date
    elif preset == "Year to date":
        start_date = datetime(max_date.year, 1, 1).date()
        end_date = max_date
    elif preset == "All time":
        start_date = min_date
        end_date = max_date
    else:
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        start_date, end_date = date_range

    messages = [
        m for m in messages
        if m.timestamp and start_date <= m.timestamp.date() <= end_date
    ]
else:
    st.sidebar.info("No timestamps detected, date filter disabled.")

messages = messages[-max_messages:]

media_results: List[MediaResult] = []
if include_media and media_root:
    media_index = _build_media_index(media_root)
    messages, media_results = _messages_with_media_text(
        messages,
        media_index=media_index,
        include_media=include_media,
        whisper_model=whisper_model,
        max_media=max_media,
        max_chars=max_media_chars,
    )
elif include_media and not media_root:
    st.warning("Media processing requires a .zip export (with media). Upload the zip to include media.")

st.subheader("Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Messages", len(messages))
with col2:
    authors = [m.author for m in messages if m.author]
    st.metric("Participants", len(set(authors)))
with col3:
    last_ts = next((m.timestamp for m in reversed(messages) if m.timestamp), None)
    st.metric("Last message", last_ts.strftime("%Y-%m-%d %H:%M") if last_ts else "Unknown")

if media_root:
    total_attachments = sum(len(m.attachments) for m in messages)
    st.caption(f"Attachments referenced: {total_attachments} (processing up to {max_media}).")

st.subheader("Stock Mentions (Heuristic)")
all_tickers = []
for m in messages:
    all_tickers.extend(_extract_tickers(m.text))

counts = Counter(all_tickers)
if counts:
    df = pd.DataFrame(counts.most_common(25), columns=["Ticker", "Mentions"])
    st.dataframe(df, use_container_width=True)
else:
    st.write("No obvious ticker symbols found.")

scores = _heuristic_sentiment(messages)
if scores:
    st.subheader("Buy/Sell Signals (Heuristic)")
    rows = []
    for t, s in scores.items():
        rows.append({"Ticker": t, "Buy": s["buy"], "Sell": s["sell"], "Hold": s["hold"]})
    st.dataframe(pd.DataFrame(rows).sort_values(by=["Buy", "Sell"], ascending=False), use_container_width=True)

st.subheader("Summary")
if use_ollama:
    if _ollama_available():
        with st.spinner("Analyzing with Ollama..."):
            try:
                summary, stocks = _ollama_insights(messages, model_name)
                st.write(summary)
                if stocks:
                    st.subheader("Stocks and Reasons (Ollama)")
                    st.dataframe(pd.DataFrame(stocks), use_container_width=True)
            except requests.RequestException as e:
                st.error(f"Ollama error: {e}")
                st.info("Showing heuristic summary instead.")
    else:
        st.warning("Ollama not detected. Install it or uncheck 'Use Ollama'.")

if not use_ollama or not _ollama_available():
    st.markdown(_heuristic_summary(messages))

if media_results:
    st.subheader("Media Processing Results")
    rows = []
    for r in media_results:
        preview = ""
        if r.text:
            preview = " ".join(r.text.split())
            if len(preview) > 220:
                preview = preview[:217] + "..."
        rows.append(
            {
                "File": os.path.basename(r.path),
                "Status": r.status,
                "Chars": len(r.text or ""),
                "Summary": preview,
                "Error": r.error or "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("Recent Messages")
preview = [
    {
        "Time": m.timestamp.strftime("%Y-%m-%d %H:%M") if m.timestamp else "",
        "Author": m.author or "",
        "Text": m.text,
    }
    for m in messages[-50:]
]
st.dataframe(pd.DataFrame(preview), use_container_width=True)
