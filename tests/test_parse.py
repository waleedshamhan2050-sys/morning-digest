#!/usr/bin/env python3
"""
Parser tests for Morning Digest — NO model required.

Run with plain Python (no pytest needed):
    python3 tests/test_parse.py
Also works under pytest (test_* functions).
"""

import importlib.util
import os
import sys
from datetime import datetime

# Import digest.py from the project root (one level up from tests/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("digest", os.path.join(ROOT, "digest.py"))
digest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(digest)


def _msg(id_, date, unix, frm, text, mtype="message"):
    m = {"id": id_, "type": mtype, "date": date, "date_unixtime": str(unix),
         "from": frm, "text": text}
    return m


# Window anchor: latest message = 2026-10-20 07:00 → 12h window starts 2026-10-19 19:00
SAMPLE_EXPORT = {
    "name": "Test Chat",
    "type": "public_supergroup",
    "id": 42,
    "messages": [
        _msg(1, "2026-10-18T10:00:00", 1792317600, "Ali", "old message, way outside"),
        _msg(2, "2026-10-19T18:59:00", 1792436340, "Ali", "just before the window"),
        _msg(3, "2026-10-19T19:00:00", 1792436400, "Sara", "exactly at window start"),
        _msg(4, "2026-10-19T23:30:00", 1792452600, "Ali", "late night message"),
        _msg(5, "2026-10-20T07:00:00", 1792479600, "Sara", "good morning"),
        # service message inside the window — must be dropped
        _msg(6, "2026-10-19T20:00:00", 1792440000, "Ali", "", mtype="service"),
        # text as entity list — must be flattened and kept
        _msg(7, "2026-10-19T21:00:00", 1792443600, "Omar",
             [{"type": "plain", "text": "hello "},
              {"type": "bold", "text": "world"}]),
        # empty text (photo with no caption) — must be dropped
        _msg(8, "2026-10-19T22:00:00", 1792447200, "Omar", ""),
    ],
}


def test_parser_keeps_only_window_messages():
    messages = digest.parse_telegram_export(SAMPLE_EXPORT)
    kept, start, end = digest.filter_window(messages, hours=12)
    texts = [m["text"] for m in kept]
    assert end == datetime(2026, 10, 20, 7, 0)
    assert start == datetime(2026, 10, 19, 19, 0)
    assert "exactly at window start" in texts
    assert "late night message" in texts
    assert "good morning" in texts
    assert "just before the window" not in texts
    assert "old message, way outside" not in texts
    assert len(kept) == 4, f"expected 4 kept messages, got {len(kept)}: {texts}"


def test_parser_flattens_entities_and_drops_service_and_empty():
    messages = digest.parse_telegram_export(SAMPLE_EXPORT)
    texts = [m["text"] for m in messages]
    assert "hello world" in texts          # entity list flattened
    assert "" not in texts                  # empty/photo messages dropped
    assert len(messages) == 6               # 8 raw - 1 service - 1 empty


def test_parser_keeps_chat_and_sender():
    messages = digest.parse_telegram_export(SAMPLE_EXPORT)
    first = messages[0]
    assert first["chat"] == "Test Chat"
    assert first["sender"] == "Ali"
    assert first["timestamp"] == datetime(2026, 10, 18, 10, 0)


def test_multi_chat_export():
    data = {"chats": {"list": [SAMPLE_EXPORT, SAMPLE_EXPORT]}}
    messages = digest.parse_telegram_export(data)
    assert len(messages) == 12


def test_text_log_parser():
    log = (
        "19.10.2026 20:15 - Ali: hello there\n"
        "19.10.2026 20:16 - Sara: general kenobi\n"
        "continued line\n"
    )
    messages = digest.parse_text_log(log)
    assert len(messages) == 2
    assert messages[0]["sender"] == "Ali"
    assert "continued line" in messages[1]["text"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
