import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from io import StringIO
from urllib.parse import urlparse
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
    "GDP", "EPS", "PM", "AM", "BSE", "TRI",
}


def _extract_tickers(text: str, allowlist: set | None = None) -> List[str]:
    found = []
    upper = text.upper()
    skip_tokens = set()
    if "BSE 500 TRI" in upper:
        skip_tokens.update({"BSE", "TRI"})
    if "MCX GOLD" in upper:
        skip_tokens.update({"MCX", "GOLD"})
    for tok in TICKER_RE.findall(text):
        raw = tok.replace("$", "").upper()
        if raw in STOP_TOKENS:
            continue
        if raw in skip_tokens:
            continue
        if len(raw) == 1:
            continue
        if allowlist is not None and raw not in allowlist:
            continue
        found.append(raw)
    return found


def _heuristic_sentiment(messages: List[Message], allowlist: set | None = None) -> Dict[str, Dict[str, int]]:
    scores: Dict[str, Dict[str, int]] = defaultdict(lambda: {"buy": 0, "sell": 0, "hold": 0})
    for msg in messages:
        text = msg.text.lower()
        tickers = _extract_tickers(msg.text, allowlist=allowlist)
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

@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_nse_symbols() -> tuple[set, Dict[str, str]]:
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/csv,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    symbols = set(str(s).strip().upper() for s in df.get("SYMBOL", []) if str(s).strip())
    name_col = "NAME OF COMPANY"
    if name_col not in df.columns:
        name_col = "NAME_OF_COMPANY" if "NAME_OF_COMPANY" in df.columns else name_col
    name_map: Dict[str, str] = {}
    if name_col in df.columns and "SYMBOL" in df.columns:
        for _, row in df.iterrows():
            sym = str(row.get("SYMBOL", "")).strip().upper()
            name = str(row.get(name_col, "")).strip()
            if sym and name:
                name_map[sym] = name
    return symbols, name_map


def _heuristic_summary(messages: List[Message], allowlist: set | None = None, top_n: int = 6) -> str:
    tickers = []
    for m in messages:
        tickers.extend(_extract_tickers(m.text, allowlist=allowlist))
    ticker_counts = Counter(tickers).most_common(top_n)

    authors = [m.author for m in messages if m.author]
    top_authors = Counter(authors).most_common(5)

    scores = _heuristic_sentiment(messages, allowlist=allowlist)
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

    # Build per-ticker reasons from nearby phrases
    reason_map: Dict[str, List[str]] = defaultdict(list)
    reason_keywords = [
        "why", "because", "due to", "as ", "since ", "on earnings", "results",
        "guidance", "margin", "revenue", "profit", "loss", "breakout",
        "support", "resistance", "rsi", "divergent", "volume", "trend",
        "target", "stop", "entry", "exit",
    ]
    for m in messages:
        text = m.text.replace("\n", " ")
        tickers_here = _extract_tickers(text, allowlist=allowlist)
        if not tickers_here:
            continue
        for kw in reason_keywords:
            if kw in text.lower():
                snippet = text
                if len(snippet) > 220:
                    snippet = snippet[:217] + "..."
                for t in tickers_here:
                    if len(reason_map[t]) < 3:
                        reason_map[t].append(snippet)
                break

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
    if reason_map:
        lines = []
        for t, snippets in reason_map.items():
            joined = " / ".join(snippets[:2])
            lines.append(f"{t}: {joined}")
        parts.append("Reasons (heuristic): " + " | ".join(lines[:top_n]))
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
def _extract_media_cached(path: str, whisper_model: str, ocr_lang: str, pdf_ocr_pages: int) -> Dict[str, str]:
    res = extract_text(
        path,
        whisper_model=whisper_model or None,
        ocr_lang=ocr_lang,
        pdf_ocr_pages=pdf_ocr_pages,
    )
    return {
        "path": res.path,
        "status": res.status,
        "text": res.text,
        "error": res.error or "",
    }


def _messages_with_media_text(
    messages: List[Message],
    media_index: Dict[str, str],
    include_media: bool,
    whisper_model: str,
    ocr_lang: str,
    pdf_ocr_pages: int,
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
            res_dict = _extract_media_cached(path, whisper_model, ocr_lang, pdf_ocr_pages)
            res = MediaResult(
                path=res_dict["path"],
                status=res_dict["status"],
                text=res_dict["text"],
                error=res_dict.get("error") or None,
            )
            results.append(res)
            if res.text:
                snippet = res.text[:max_chars].strip()
                if snippet:
                    msg.text = (msg.text + f"\n[Media {name}] {snippet}").strip()
            processed += 1
        if processed >= max_media:
            break

    return messages, results


def _date_from_filename(name: str) -> datetime | None:
    # Matches WhatsApp media names with embedded timestamps.
    m = re.search(r"(20\\d{2})[-_](\\d{2})[-_](\\d{2})[-_](\\d{2})[-_](\\d{2})[-_](\\d{2})", name)
    if not m:
        return None
    try:
        y, mo, d, h, mi, s = (int(x) for x in m.groups())
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


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

uploaded = st.file_uploader("Upload WhatsApp chat export (.txt) or zip (with media)", type=["txt", "zip"])

if uploaded is None:
    st.info("Upload a WhatsApp exported chat file to begin.")
    st.stop()
    raise SystemExit

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

file_id = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
if st.session_state.get("last_file_id") != file_id:
    st.session_state["last_file_id"] = file_id
    st.session_state["confirmed"] = False
    st.session_state["auto_defaults_done"] = False

lines = chat_text.splitlines()
messages = filter_system_messages(parse_chat(lines))

if not messages:
    st.error("No messages parsed. Make sure this is a WhatsApp exported chat text file.")
    st.stop()

url_re = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?)", re.IGNORECASE)
http_re = re.compile(r"https?://\S+", re.IGNORECASE)
def _normalize_for_links(text: str) -> str:
    # Remove common invisible markers and normalize whitespace.
    for ch in ["\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069"]:
        text = text.replace(ch, "")
    text = text.replace("\u202f", " ")
    text = text.replace("\u00a0", " ")
    return " ".join(text.split())

all_urls = []
clean_text = _normalize_for_links(chat_text)
for raw in url_re.findall(clean_text):
    # Filter out emails
    if "@" in raw and "://" not in raw:
        continue
    # Trim trailing punctuation
    cleaned = raw.rstrip(").,;]>\"'")
    all_urls.append(cleaned)

http_urls = []
for raw in http_re.findall(clean_text):
    cleaned = raw.rstrip(").,;]>\"'")
    http_urls.append(cleaned)

if st.checkbox("Show detected URLs (debug)", value=False):
    http_count = clean_text.lower().count("http")
    st.write(
        {
            "http_count": http_count,
            "first_http_urls": http_urls[:20],
            "first_all_urls": all_urls[:20],
        }
    )
domain_counts = Counter()
for u in all_urls:
    try:
        host = urlparse(u).netloc.lower()
    except Exception:
        continue
    host = host.replace("www.", "")
    if host:
        domain_counts[host] += 1

http_domain_counts = Counter()
for u in http_urls:
    try:
        host = urlparse(u).netloc.lower()
    except Exception:
        continue
    host = host.replace("www.", "")
    if host:
        http_domain_counts[host] += 1

media_counts = Counter()
if media_root:
    for root, _, files in os.walk(media_root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext:
                media_counts[ext] += 1

if media_root and not st.session_state.get("auto_defaults_done"):
    total_media = sum(media_counts.values())
    audio_video = sum(media_counts[e] for e in [".mp3", ".m4a", ".wav", ".aac", ".mp4", ".mov", ".mkv"] if e in media_counts)
    st.session_state["include_media"] = total_media > 0
    st.session_state["max_media"] = min(300, max(20, total_media)) if total_media else 80
    st.session_state["media_date_source_idx"] = 1  # Filename date
    st.session_state["whisper_model_idx"] = 3 if audio_video else 2  # small if AV, else base
    st.session_state["pdf_ocr_pages"] = 8 if media_counts.get(".pdf") else 5
    st.session_state["max_media_chars"] = 2000
    st.session_state["validate_symbols"] = True
    st.session_state["auto_defaults_done"] = True
    st.rerun()

with st.sidebar:
    st.header("Settings")
    use_ollama = st.checkbox("Use Ollama (local LLM)", value=st.session_state.get("use_ollama", True), key="use_ollama")
    model_name = st.text_input("Ollama model", value=st.session_state.get("model_name", "llama3.1"), key="model_name")
    max_messages = st.slider(
        "Messages to analyze",
        min_value=200,
        max_value=5000,
        value=st.session_state.get("max_messages", 1200),
        step=200,
        key="max_messages",
    )
    validate_symbols = st.checkbox("Validate tickers with live NSE list", value=st.session_state.get("validate_symbols", True), key="validate_symbols")
    include_media = st.checkbox("Include media OCR/transcripts", value=st.session_state.get("include_media", True), key="include_media")
    ocr_lang = st.text_input("OCR language (Tesseract)", value=st.session_state.get("ocr_lang", "eng"), key="ocr_lang")
    pdf_ocr_pages = st.slider(
        "PDF OCR pages (if scanned)",
        min_value=1,
        max_value=20,
        value=st.session_state.get("pdf_ocr_pages", 5),
        key="pdf_ocr_pages",
    )
    max_media = st.slider(
        "Max media files to process",
        min_value=10,
        max_value=500,
        value=st.session_state.get("max_media", 80),
        step=10,
        key="max_media",
    )
    max_media_chars = st.slider(
        "Max chars per media item",
        min_value=200,
        max_value=4000,
        value=st.session_state.get("max_media_chars", 1200),
        step=200,
        key="max_media_chars",
    )
    whisper_model = st.selectbox(
        "Audio model (whisper)",
        ["", "tiny", "base", "small", "medium"],
        index=st.session_state.get("whisper_model_idx", 1),
        key="whisper_model",
    )
    media_date_source = st.selectbox(
        "Media date source",
        ["Chat timestamp", "Filename date", "File modified time"],
        index=st.session_state.get("media_date_source_idx", 0),
        key="media_date_source",
    )

st.subheader("Preflight Summary")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Messages", len(messages))
with col_b:
    st.metric("Links (all)", len(all_urls))
with col_c:
    st.metric("Media files", sum(media_counts.values()) if media_root else 0)

col_d, col_e, col_f = st.columns(3)
with col_d:
    st.metric("Links (http)", len(http_urls))
with col_e:
    st.metric("Unique links (http)", len(set(http_urls)))
with col_f:
    st.metric("Unique domains (http)", len(http_domain_counts))

if domain_counts:
    st.markdown("**Top link domains (all links)**")
    df_all_domains = pd.DataFrame(domain_counts.most_common(15), columns=["Domain", "Count"])
    df_all_domains.insert(0, "#", range(1, len(df_all_domains) + 1))
    st.dataframe(df_all_domains, use_container_width=True, hide_index=True)
if http_domain_counts:
    st.markdown("**Top link domains (http only)**")
    df_http_domains = pd.DataFrame(http_domain_counts.most_common(15), columns=["Domain", "Count"])
    df_http_domains.insert(0, "#", range(1, len(df_http_domains) + 1))
    st.dataframe(df_http_domains, use_container_width=True, hide_index=True)

if media_root and media_counts:
    st.markdown("**Media type counts**")
    df_media_counts = pd.DataFrame(media_counts.most_common(), columns=["Extension", "Count"])
    df_media_counts.insert(0, "#", range(1, len(df_media_counts) + 1))
    st.dataframe(df_media_counts, use_container_width=True, hide_index=True)

if not st.session_state.get("confirmed"):
    if st.button("Continue to Analysis"):
        st.session_state["confirmed"] = True
        st.rerun()
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

symbol_allowlist: set | None = None
symbol_names: Dict[str, str] = {}
if validate_symbols:
    try:
        symbol_allowlist, symbol_names = _fetch_nse_symbols()
        st.sidebar.caption(f"NSE symbols loaded: {len(symbol_allowlist)}")
    except requests.RequestException as e:
        st.sidebar.warning(f"Could not fetch NSE list. Using heuristic tickers. ({e})")
        symbol_allowlist = None
        symbol_names = {}

media_results: List[MediaResult] = []
attachment_times: Dict[str, datetime] = {}
for m in messages:
    if m.timestamp:
        for name in m.attachments:
            attachment_times.setdefault(name, m.timestamp)

if include_media and media_root:
    media_index = _build_media_index(media_root)
    messages, media_results = _messages_with_media_text(
        messages,
        media_index=media_index,
        include_media=include_media,
        whisper_model=whisper_model,
        ocr_lang=ocr_lang,
        pdf_ocr_pages=pdf_ocr_pages,
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
    all_tickers.extend(_extract_tickers(m.text, allowlist=symbol_allowlist))

counts = Counter(all_tickers)
if counts:
    df = pd.DataFrame(counts.most_common(25), columns=["Ticker", "Mentions"])
    if symbol_names:
        df.insert(1, "Company", df["Ticker"].map(symbol_names).fillna(""))
    st.dataframe(df, use_container_width=True)
else:
    st.write("No obvious ticker symbols found.")

scores = _heuristic_sentiment(messages, allowlist=symbol_allowlist)
if scores:
    st.subheader("Buy/Sell Signals (Heuristic)")
    rows = []
    for t, s in scores.items():
        row = {"Ticker": t, "Buy": s["buy"], "Sell": s["sell"], "Hold": s["hold"]}
        if symbol_names:
            row["Company"] = symbol_names.get(t, "")
        rows.append(row)
    df_scores = pd.DataFrame(rows)
    if "Company" in df_scores.columns:
        df_scores = df_scores[["Ticker", "Company", "Buy", "Sell", "Hold"]]
    st.dataframe(df_scores.sort_values(by=["Buy", "Sell"], ascending=False), use_container_width=True)

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
    st.markdown(_heuristic_summary(messages, allowlist=symbol_allowlist))

if media_results:
    st.subheader("Media Processing Results")
    rows = []
    for r in media_results:
        preview = ""
        if r.text:
            preview = " ".join(r.text.split())
            if len(preview) > 220:
                preview = preview[:217] + "..."
        name = os.path.basename(r.path)
        ts = None
        if media_date_source == "Chat timestamp":
            ts = attachment_times.get(name)
        elif media_date_source == "Filename date":
            ts = _date_from_filename(name)
        elif media_date_source == "File modified time":
            try:
                ts = datetime.fromtimestamp(os.path.getmtime(r.path))
            except OSError:
                ts = None
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else ""
        rows.append(
            {
                "File": name,
                "Date": ts_str,
                "Status": r.status,
                "Chars": len(r.text or ""),
                "Summary": preview,
                "Error": r.error or "",
            }
        )
    df_media = pd.DataFrame(rows)
    if "Date" in df_media.columns and not df_media.empty:
        df_media = df_media.sort_values(by=["Date"], ascending=False, na_position="last")
    df_media.insert(0, "#", range(1, len(df_media) + 1))
    table_html = df_media.to_html(index=False, escape=True)
    st.markdown(
        """
        <style>
        .media-table table {
            width: 100%;
            border-collapse: collapse;
        }
        .media-table th, .media-table td {
            border: 1px solid #e6e6e6;
            padding: 6px 8px;
            vertical-align: top;
        }
        .media-table td {
            white-space: normal;
            word-break: break-word;
            max-width: 520px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='media-table'>{table_html}</div>", unsafe_allow_html=True)

st.subheader("Recent Messages")
sorted_messages = sorted(
    messages,
    key=lambda m: m.timestamp or datetime.min,
    reverse=True,
)
preview = [
    {
        "Time": m.timestamp.strftime("%Y-%m-%d %H:%M") if m.timestamp else "",
        "Author": m.author or "",
        "Text": m.text,
    }
    for m in sorted_messages[:50]
]
df_preview = pd.DataFrame(preview)
table_html = df_preview.to_html(index=False, escape=True)
st.markdown(
    """
    <style>
    .recent-table table {
        width: 100%;
        border-collapse: collapse;
    }
    .recent-table th, .recent-table td {
        border: 1px solid #e6e6e6;
        padding: 6px 8px;
        vertical-align: top;
    }
    .recent-table td {
        white-space: normal;
        word-break: break-word;
        max-width: 520px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(f"<div class='recent-table'>{table_html}</div>", unsafe_allow_html=True)
