#!/usr/bin/env python3
"""
Morning Digest — a local agent that reads your chat exports overnight and
hands you a short Arabic digest in the morning.

Built for the Hacktoberfest Open-Source AI Challenge Week 3: "Slow and Steady"
(build something with open-weight models that rewards laziness).

It reads a Telegram Desktop export (result.json) or a plain-text chat log,
takes the last N hours of messages, and summarizes them locally with
Gemma 3 via Ollama — no cloud, no API key, your chats never leave your machine.
An optional OpenAI provider is available (--provider openai), following the
same provider pattern as the AI Post Generator project.

Usage:
    python3 digest.py result.json                 # last 12 hours, Ollama/Gemma 3
    python3 digest.py chat.txt --hours 24
    python3 digest.py result.json --provider openai

Output: printed to the terminal AND saved to digest-YYYY-MM-DD.md
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta

import requests

# ---------------------------------------------------------------------------
# Minimal .env loader (keeps requirements.txt to just `requests`)
# ---------------------------------------------------------------------------

def load_dotenv(path=".env"):
    """Load KEY=VALUE lines from a .env file into os.environ (no override)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Parsing — Telegram Desktop JSON export
# ---------------------------------------------------------------------------

def _flatten_text(text):
    """Telegram exports message text as a string OR a list of entities."""
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return ""


def _parse_tg_datetime(msg):
    """Return a datetime for a Telegram message.

    Prefer the ISO `date` field: it is the wall-clock time the export itself
    shows (the chat's local time), so window filtering behaves the same in
    every machine timezone. Fall back to `date_unixtime` (absolute time,
    converted to local time) only when `date` is missing or unparseable.
    """
    raw = msg.get("date")
    if raw:
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    unix = msg.get("date_unixtime")
    if unix:
        try:
            return datetime.fromtimestamp(int(unix))
        except (ValueError, OSError):
            return None
    return None


def parse_telegram_export(data):
    """
    Parse a Telegram Desktop export into a flat message list.

    Supports both a single-chat export ({"name": ..., "messages": [...]})
    and a full export ({"chats": {"list": [ {chat}, ... ]}}).

    Returns a list of dicts: {timestamp, chat, sender, text}.
    Service messages and empty texts are dropped.
    """
    chats = []
    if isinstance(data, dict) and "messages" in data:
        chats = [data]
    elif isinstance(data, dict):
        chats = (data.get("chats") or {}).get("list") or []

    messages = []
    for chat in chats:
        chat_name = chat.get("name") or "محادثة"
        for msg in chat.get("messages", []):
            if msg.get("type") != "message":
                continue  # skip service events (joins, pins, calls...)
            text = _flatten_text(msg.get("text", "")).strip()
            if not text:
                continue
            ts = _parse_tg_datetime(msg)
            if ts is None:
                continue
            messages.append({
                "timestamp": ts,
                "chat": chat_name,
                "sender": msg.get("from") or "مجهول",
                "text": text,
            })
    messages.sort(key=lambda m: m["timestamp"])
    return messages


# ---------------------------------------------------------------------------
# Parsing — plain-text chat log
# ---------------------------------------------------------------------------

# Tolerates common export shapes, e.g.:
#   19.10.2026 08:15 - Waleed: hello
#   10/19/2026, 8:15 PM - Sara: hi
#   2026-10-19 08:15:00 Waleed: hello
_TXT_LINE_RE = re.compile(
    r"^(?:"
    r"(?P<d1>\d{1,2})[./](?P<m1>\d{1,2})[./](?P<y1>\d{2,4})"          # D/M/Y
    r"|(?P<y2>\d{4})-(?P<m2>\d{1,2})-(?P<d2>\d{1,2})"                # Y-M-D
    r")[ ,T]+"
    r"(?P<h>\d{1,2}):(?P<min>\d{2})(?::(?P<s>\d{2}))?\s*(?P<ampm>AM|PM|am|pm)?"
    r"\s*[-–—]?\s*"
    r"(?P<sender>[^:\n]{1,60}):\s*(?P<text>.*)$"
)


def parse_text_log(content):
    """
    Parse a plain-text chat log into the same flat message list.
    Lines that don't start a new message are treated as continuations.
    """
    messages = []
    for line in content.splitlines():
        m = _TXT_LINE_RE.match(line.strip())
        if not m:
            if messages:
                messages[-1]["text"] += "\n" + line
            continue
        g = m.groupdict()
        if g["y2"]:
            year, month, day = int(g["y2"]), int(g["m2"]), int(g["d2"])
        else:
            year = int(g["y1"])
            if year < 100:
                year += 2000
            day, month = int(g["d1"]), int(g["m1"])
        hour = int(g["h"])
        if g["ampm"]:
            if g["ampm"].lower() == "pm" and hour < 12:
                hour += 12
            elif g["ampm"].lower() == "am" and hour == 12:
                hour = 0
        try:
            ts = datetime(year, month, day, hour, int(g["min"]), int(g["s"] or 0))
        except ValueError:
            continue
        messages.append({
            "timestamp": ts,
            "chat": "محادثة نصية",
            "sender": g["sender"].strip(),
            "text": g["text"].strip(),
        })
    messages.sort(key=lambda m: m["timestamp"])
    return messages


def load_messages(path):
    """Load messages from a .json (Telegram export) or .txt chat log."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if path.lower().endswith(".json") or content.lstrip().startswith("{"):
        return parse_telegram_export(json.loads(content))
    return parse_text_log(content)


def filter_window(messages, hours):
    """
    Keep only the messages from the last `hours` hours of activity.
    The window is anchored at the latest message in the file (exports are
    usually generated in the morning, so 'recent' = the tail of the file).
    Returns (kept_messages, window_start, window_end).
    """
    if not messages:
        return [], None, None
    end = max(m["timestamp"] for m in messages)
    start = end - timedelta(hours=hours)
    kept = [m for m in messages if start <= m["timestamp"] <= end]
    return kept, start, end


# ---------------------------------------------------------------------------
# Model providers (requests only — Ollama native API / OpenAI chat API)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "أنت مساعد يلخّص محادثات الدردشة بدقة. لخّص بالعربية الفصيحة المبسطة، "
    "ولا تخترع معلومات غير موجودة في النص، ولا تضف آراء من عندك."
)


def _ollama_base_url():
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    base = base.rstrip("/")
    if base.endswith("/v1"):  # tolerate the OpenAI-style URL from .env
        base = base[:-3]
    return base


def call_ollama(prompt, model):
    """One chat completion against a local Ollama server (native API)."""
    # Localhost must never go through a proxy (mirrors the AI Post Generator fix).
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)
    url = _ollama_base_url() + "/api/chat"
    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=600,
            proxies={"http": None, "https": None},
        )
    except requests.ConnectionError:
        raise RuntimeError(
            f"Could not reach Ollama at {_ollama_base_url()}.\n"
            "Is the server running? Start it with:  ollama serve\n"
            f"And make sure the model is pulled:  ollama pull {model}"
        )
    except requests.Timeout:
        raise RuntimeError("Ollama took too long to answer (timeout). "
                           "Try a smaller model or a shorter --hours window.")
    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama answered but model '{model}' was not found. "
            f"Pull it first:  ollama pull {model}"
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["message"]["content"].strip()


def call_openai(prompt, model):
    """One chat completion against the OpenAI API (optional provider)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or use "
                           "--provider ollama (default, fully local).")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    try:
        resp = requests.post(
            base + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=300,
        )
    except requests.ConnectionError:
        raise RuntimeError("Could not reach the OpenAI API. Check your connection.")
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_model(prompt, provider, model):
    if provider == "ollama":
        return call_ollama(prompt, model)
    return call_openai(prompt, model)


# ---------------------------------------------------------------------------
# Summarization (chunked: summarize chunks, then summarize the summaries)
# ---------------------------------------------------------------------------

def format_messages(messages):
    lines = []
    for m in messages:
        lines.append(f"[{m['timestamp']:%H:%M}] ({m['chat']}) {m['sender']}: {m['text']}")
    return "\n".join(lines)


def chunk_text(text, max_chars=6000):
    """Split formatted messages into chunks on line boundaries."""
    chunks, current = [], ""
    for line in text.splitlines():
        if len(current) + len(line) + 1 > max_chars and current:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


def summarize(messages, provider, model, max_chunk_chars=6000):
    """Two-stage summarization: per-chunk notes, then the final digest."""
    formatted = format_messages(messages)
    chunks = chunk_text(formatted, max_chunk_chars)

    chunk_notes = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"… summarizing chunk {i}/{len(chunks)}", file=sys.stderr)
        prompt = (
            "هذه رسائل من محادثات دردشة. لخّصها في نقاط عربية موجزة: "
            "المواضيع التي نوقشت، وأي قرارات أو مهام أو مواعيد ذُكرت. "
            "حافظ على أسماء الأشخاص كما هي.\n\n" + chunk
        )
        chunk_notes.append(call_model(prompt, provider, model))

    if len(chunk_notes) == 1:
        notes = chunk_notes[0]
    else:
        print("… combining chunk summaries", file=sys.stderr)
        notes = "\n\n".join(chunk_notes)

    final_prompt = (
        "هذه ملخصات لمحادثات جرت خلال الليل. اكتب «ملخص الصباح» النهائي "
        "بالعربية بهذا الشكل بالضبط:\n\n"
        "سطر عنوان رئيسي واحد يلخّص أهم ما حدث (لا يتجاوز سطرين).\n\n"
        "أهم المواضيع:\n"
        "- نقطة لكل موضوع مهم\n\n"
        "قرارات ومهام:\n"
        "- أي قرار اتُّخذ أو مهمة أو موعد تم الاتفاق عليه\n"
        "إذا لم توجد قرارات أو مهام واضحة اكتب تحت هذا العنوان: لا توجد.\n\n"
        "الملخصات:\n" + notes
    )
    return call_model(final_prompt, provider, model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Morning Digest — summarize last night's chats locally with Gemma 3 + Ollama")
    parser.add_argument("input", help="Telegram export (result.json) or plain-text chat log (.txt)")
    parser.add_argument("--hours", type=float, default=12,
                        help="Recent window in hours, anchored at the latest message (default: 12)")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama",
                        help="'ollama' = local open-weight model (default), 'openai' = hosted API")
    parser.add_argument("--model", default=None,
                        help="Model name (default: OLLAMA_MODEL env or 'gemma3'; 'gpt-4.1-mini' for openai)")
    parser.add_argument("--out", default=None,
                        help="Output file (default: digest-YYYY-MM-DD.md)")
    args = parser.parse_args()

    if args.provider == "ollama":
        model = args.model or os.getenv("OLLAMA_MODEL", "gemma3")
    else:
        model = args.model or "gpt-4.1-mini"

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        return 1

    try:
        messages = load_messages(args.input)
    except json.JSONDecodeError:
        print(f"Error: {args.input} is not valid JSON. "
              "Export from Telegram Desktop as JSON, or use a .txt chat log.")
        return 1

    kept, start, end = filter_window(messages, args.hours)
    if not kept:
        print("No messages found in the selected window — nothing to summarize. "
              "Try a larger --hours value.")
        return 0

    print(f"{len(kept)} messages | window {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} "
          f"| provider {args.provider} | model {model}", file=sys.stderr)

    try:
        digest_body = summarize(kept, args.provider, model)
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    out_path = args.out or f"digest-{end:%Y-%m-%d}.md"
    document = (
        f"# ☀️ ملخص الصباح — {end:%Y-%m-%d}\n\n"
        f"{digest_body}\n\n"
        f"---\n"
        f"*{len(kept)} رسالة من {start:%d %b %H:%M} إلى {end:%d %b %H:%M} "
        f"| النموذج: {model} ({args.provider}) — يعمل محليًا*\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(document)

    print(document)
    print(f"Saved to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
