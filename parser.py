import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional


@dataclass
class Message:
    timestamp: Optional[datetime]
    author: Optional[str]
    text: str
    attachments: List[str] = field(default_factory=list)


# Handles both:
# 12/31/23, 10:05 PM - Name: message
# 31/12/23, 22:05 - Name: message
# 12/31/2023, 10:05 PM - Name: message
# 31/12/2023, 22:05 - Name: message
LINE_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s(?P<time>\d{1,2}:\d{2}(?:\s?[APMapm]{2})?)\s-\s(?P<body>.*)$"
)
BRACKET_RE = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s(?P<time>\d{1,2}:\d{2}:\d{2}(?:\s?[APMapm]{2})?)\]\s(?P<body>.*)$"
)
ATTACH_RE = re.compile(r"<attached:\s*([^>]+)>")


def _normalize_line(line: str) -> str:
    # WhatsApp exports often include invisible RTL/LTR markers.
    return line.replace("\u200e", "").replace("\u202f", " ")


def _parse_datetime(date_str: str, time_str: str, day_first: Optional[bool] = None) -> Optional[datetime]:
    time_str = time_str.strip()
    month_first = [
        "%m/%d/%y %I:%M %p",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%y %H:%M",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %I:%M:%S %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]
    day_first_list = [
        "%d/%m/%y %H:%M",
        "%d/%m/%Y %H:%M",
        "%d/%m/%y %I:%M %p",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%y %I:%M:%S %p",
        "%d/%m/%Y %I:%M:%S %p",
    ]
    if day_first is True:
        candidates = day_first_list + month_first
    elif day_first is False:
        candidates = month_first + day_first_list
    else:
        candidates = month_first + day_first_list
    for fmt in candidates:
        try:
            return datetime.strptime(f"{date_str} {time_str}", fmt)
        except ValueError:
            continue
    return None


def _detect_day_first(lines: Iterable[str]) -> Optional[bool]:
    for raw in lines:
        line = _normalize_line(raw.rstrip("\n"))
        m = LINE_RE.match(line) or BRACKET_RE.match(line)
        if not m:
            continue
        date_str = m.group("date")
        parts = date_str.split("/")
        if len(parts) != 3:
            continue
        try:
            first = int(parts[0])
            second = int(parts[1])
        except ValueError:
            continue
        if first > 12 and second <= 12:
            return True
        if second > 12 and first <= 12:
            return False
    return None


def parse_chat(lines: Iterable[str]) -> List[Message]:
    lines = list(lines)
    messages: List[Message] = []
    current: Optional[Message] = None
    day_first = _detect_day_first(lines)

    for raw in lines:
        line = _normalize_line(raw.rstrip("\n"))
        if not line:
            continue
        m = LINE_RE.match(line) or BRACKET_RE.match(line)
        if m:
            date_str = m.group("date")
            time_str = m.group("time")
            body = m.group("body")

            author = None
            text = body
            if ": " in body:
                author, text = body.split(": ", 1)

            attachments = ATTACH_RE.findall(text)
            if attachments:
                # Remove attachment markers from visible text
                text = ATTACH_RE.sub("", text).strip()

            current = Message(
                timestamp=_parse_datetime(date_str, time_str, day_first=day_first),
                author=author,
                text=text,
                attachments=attachments,
            )
            messages.append(current)
        else:
            # Continuation of previous message
            if current is not None:
                current.text += "\n" + line
            else:
                # Unstructured line, keep it as a system message
                messages.append(Message(timestamp=None, author=None, text=line))

    return messages


def filter_system_messages(messages: List[Message]) -> List[Message]:
    filtered = []
    for msg in messages:
        text = msg.text.strip()
        if text.endswith("<Media omitted>") or text.endswith("<Media omitted>"):
            continue
        if text.lower().endswith("messages and calls are end-to-end encrypted."):
            continue
        if "advanced chat privacy was turned on" in text.lower():
            continue
        filtered.append(msg)
    return filtered
