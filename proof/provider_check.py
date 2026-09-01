"""Closes the one gap the zombie proof could not measure: does a real provider
actually reject the corrupted thread?

`proof.py` shows the forked thread yields a message list containing a
tool_result whose tool_use is stored nowhere. That the shape is *invalid* was
argued from the emitted payload, never measured -- nothing installed validates
it (langchain_anthropic._format_messages, ChatAnthropic._get_request_payload
and convert_to_openai_messages all accept it happily).

This sends the real durable state -- read from the database proof.py leaves
behind, not retyped -- to a live chat-completions endpoint, and sends the
fenced thread as a control. Without the control a 4xx would only prove we
built the request badly.

Credentials are never taken from argv or the environment: point
PROVIDER_KEY_FILE at a mode-600 file. Skips cleanly when unset, so this stays
committable and re-runnable by anyone with their own key.

    PROVIDER_KEY_FILE=/path/to/key spike/.venv/bin/python proof/provider_check.py

Run proof.py first; this reads the threads it created.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "spike"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ENDPOINT = os.environ.get("PROVIDER_ENDPOINT", "https://eu.api.openai.com/v1/chat/completions")
MODEL = os.environ.get("PROVIDER_MODEL", "gpt-4o-mini")
KEY_FILE = os.environ.get("PROVIDER_KEY_FILE", "")

_KEY = ""


def scrub(s: str) -> str:
    return s.replace(_KEY, "<REDACTED>") if _KEY else s


def to_openai(messages) -> list[dict]:
    """LangChain messages -> chat-completions wire format, verbatim: no repair,
    no reordering, no dropping. The point is to send what the thread holds."""
    out: list[dict] = []
    for m in messages:
        kind = type(m).__name__
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        if kind == "HumanMessage":
            out.append({"role": "user", "content": content})
        elif kind == "AIMessage":
            calls = getattr(m, "tool_calls", None) or []
            msg: dict = {"role": "assistant", "content": content or None}
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c.get("name", "lookup"),
                                     "arguments": json.dumps(c.get("args", {}))},
                    }
                    for c in calls
                ]
            out.append(msg)
        elif kind == "ToolMessage":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": content})
        elif kind == "SystemMessage":
            out.append({"role": "system", "content": content})
    return out


def post(payload: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()[:600]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:900]


def audit(wire: list[dict]) -> list[str]:
    """Which tool_call_ids are answered by a tool message but offered by no
    preceding assistant message. Independent of the proof's own orphans()."""
    offered: set[str] = set()
    problems = []
    for m in wire:
        if m["role"] == "assistant":
            offered |= {c["id"] for c in m.get("tool_calls", [])}
        elif m["role"] == "tool" and m["tool_call_id"] not in offered:
            problems.append(m["tool_call_id"])
    return problems


def main() -> int:
    global _KEY
    if not KEY_FILE or not pathlib.Path(KEY_FILE).is_file():
        print("SKIPPED: set PROVIDER_KEY_FILE to a file containing an API key.")
        print("         Nothing was sent. This is not a failure.")
        return 0
    _KEY = pathlib.Path(KEY_FILE).read_text().strip()

    from _harness import pg_conninfo
    import proof as P

    conninfo = pg_conninfo("durable_agents_proof")
    agent = P.agent_for(conninfo)

    print(f"endpoint : {ENDPOINT}")
    print(f"model    : {MODEL}")
    print("key      : read from PROVIDER_KEY_FILE, never logged\n")

    results = {}
    for thread, label, expect in (
        ("half-a", "the CORRUPTED thread (fencing was off)", "4xx"),
        ("half-b", "the FENCED thread (control)", "2xx"),
    ):
        st = agent.get_state({"configurable": {"thread_id": thread}})
        msgs = st.values.get("messages", [])
        wire = to_openai(msgs)
        bad = audit(wire)
        print(f"=== {label}  [thread_id={thread}]")
        print(f"    {len(wire)} messages; tool_result ids offered by no preceding tool_use: {bad or 'none'}")
        status, body = post({"model": MODEL, "messages": wire, "max_tokens": 16})
        ok = (400 <= status < 500) if expect == "4xx" else (200 <= status < 300)
        results[thread] = (status, ok)
        print(f"    HTTP {status}   expected {expect}   {'as expected' if ok else 'NOT as expected'}")
        try:
            err = json.loads(body).get("error", {})
            if err:
                print(f"    provider says: {scrub(str(err.get('message','')))[:300]}")
                print(f"    code={err.get('code')} param={err.get('param')} type={err.get('type')}")
        except Exception:
            print(f"    body: {scrub(body)[:200]}")
        print()

    a_status, a_ok = results["half-a"]
    b_status, b_ok = results["half-b"]
    print("=" * 78)
    if a_ok and b_ok:
        print("MEASURED: a real provider REJECTS the corrupted thread and ACCEPTS the")
        print(f"          fenced one. half-a -> HTTP {a_status}, half-b -> HTTP {b_status}.")
        print("          The corruption is no longer argued from payload shape; it is")
        print("          a measured, reproducible provider-side failure.")
        return 0
    print(f"INCONCLUSIVE: half-a -> HTTP {a_status} (wanted 4xx), half-b -> HTTP {b_status} (wanted 2xx).")
    print("          Do not upgrade the claim in the README on this basis.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
