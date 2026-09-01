"""E5: the third _cursor branch (connection-level Pipeline, self.pipe set).
E7: the ON CONFLICT branch selection -- DO NOTHING vs DO UPDATE.
E8: does anything installed actually validate the malformed list?

Provenance: copied VERBATIM (not one character changed) from the adversarial-
review scratchpad as `e5_e7.py`. Supporting/exploratory probes, not primary
evidence for any single committed claim -- see ../README.md for how these
relate to the mutation and conflict-clause evidence. No path fix was needed:
this script already hardcodes this repo's absolute path.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spike"))
sys.path.insert(0, str(ROOT / "proof"))

import psycopg  # noqa: E402
from _harness import pg_conninfo  # noqa: E402
from langgraph.checkpoint.base import WRITES_IDX_MAP  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from langgraph.checkpoint.postgres.base import (  # noqa: E402
    INSERT_CHECKPOINT_WRITES_SQL, UPSERT_CHECKPOINT_WRITES_SQL)

from fence import FencedPostgresSaver, install_fence, setup_database  # noqa: E402

CI = pg_conninfo("e5_e7")
setup_database(CI)
print(f"WRITES_IDX_MAP = {WRITES_IDX_MAP}")

# ---------------------------------------------------------------------- E5 ---
print("\n=== E5: put_writes with a STALE fence, in each _cursor(pipeline=True) branch ===")
cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "cp1"}}


def rows(tid):
    with psycopg.connect(CI, autocommit=True) as c:
        return c.execute("select count(*) from checkpoint_writes where thread_id=%s",
                         (tid,)).fetchone()[0]


for label, mk in (
    ("self.pipe set (from_conn_string(pipeline=True) shape)", "pipe"),
    ("supports_pipeline=True (conn.pipeline() per write)", "sp_true"),
    ("supports_pipeline=False (conn.transaction() fallback)", "sp_false"),
):
    tid = f"e5-{mk}"
    install_fence(CI, tid, "A")  # fence = 1
    c = {"configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": "cp1"}}
    exc = None
    conn = psycopg.connect(CI, autocommit=True, prepare_threshold=0,
                           row_factory=psycopg.rows.dict_row)
    try:
        if mk == "pipe":
            with conn.pipeline() as pipe:
                s = FencedPostgresSaver(conn, tid, 99, pipe=pipe)  # 99 = STALE
                try:
                    s.put_writes(c, [("messages", "x")], "task-1")
                except BaseException as e:  # noqa: BLE001
                    exc = e
        else:
            s = FencedPostgresSaver(conn, tid, 99)
            s.supports_pipeline = (mk == "sp_true")
            try:
                s.put_writes(c, [("messages", "x")], "task-1")
            except BaseException as e:  # noqa: BLE001
                exc = e
    except BaseException as e:  # noqa: BLE001  (pipeline __exit__ can raise)
        exc = exc or e
    finally:
        try:
            conn.close()
        except BaseException:
            pass
    print(f"  {label}")
    print(f"      exc={type(exc).__name__ if exc else None}"
          f" sqlstate={getattr(exc, 'sqlstate', None)} msg={str(exc)[:60]!r}")
    print(f"      write refused? rows for thread = {rows(tid)}")

# ---------------------------------------------------------------------- E7 ---
print("\n=== E7: the two put_writes statements, same conflict key, twice ===")
with psycopg.connect(CI, autocommit=True) as c:
    for label, sql in (("INSERT ... DO NOTHING (any non-WRITES_IDX_MAP channel)",
                        INSERT_CHECKPOINT_WRITES_SQL),
                       ("UPSERT ... DO UPDATE (all channels in WRITES_IDX_MAP)",
                        UPSERT_CHECKPOINT_WRITES_SQL)):
        tid = "e7-" + ("nothing" if "DO NOTHING" in sql else "update")
        for who in ("survivor", "zombie"):
            c.execute(sql, (tid, "", "cp1", "task-1", "", 0, "__interrupt__",
                            "text", who.encode()))
        got = c.execute("select channel, blob from checkpoint_writes where thread_id=%s",
                        (tid,)).fetchall()
        print(f"  {label}\n      surviving row -> {[(ch, bytes(b).decode()) for ch, b in got]}")

print("\n  which statement does put_writes pick?")
for writes in ([("messages", 1)], [("__interrupt__", 1)], [("__resume__", 1)],
               [("__interrupt__", 1), ("messages", 1)]):
    chans = [w[0] for w in writes]
    pick = ("UPSERT/DO UPDATE" if all(w[0] in WRITES_IDX_MAP for w in writes)
            else "INSERT/DO NOTHING")
    print(f"      {str(chans):48} -> {pick}")

# ---------------------------------------------------------------------- E8 ---
print("\n=== E8: does anything installed reject the malformed list? ===")
from langchain_core.messages import (AIMessage, HumanMessage,  # noqa: E402
                                     ToolMessage, convert_to_openai_messages)

BAD = [HumanMessage("go"),
       AIMessage("looking up k1", tool_calls=[{"name": "lookup", "args": {"key": "k1"}, "id": "tc_A1"}]),
       ToolMessage("value-of-k1", tool_call_id="tc_A1"),
       AIMessage("worker B turn 2", tool_calls=[{"name": "lookup", "args": {"key": "k2"}, "id": "tc_B2"}]),
       ToolMessage("value-of-k2", tool_call_id="tc_A2"),
       AIMessage("final answer from A")]

import langchain_anthropic.chat_models as lacm  # noqa: E402

try:
    sysmsg, formatted = lacm._format_messages(BAD)
    print("  langchain_anthropic._format_messages  -> ACCEPTED, "
          f"{len(formatted)} blocks")
    for b in formatted:
        kinds = [(x.get("type"), x.get("id") or x.get("tool_use_id"))
                 for x in b["content"]] if isinstance(b["content"], list) else b["content"]
        print(f"      role={b['role']:<9} {kinds}")
except BaseException as e:  # noqa: BLE001
    print(f"  langchain_anthropic._format_messages  -> {type(e).__name__}: {e}")

try:
    llm = lacm.ChatAnthropic(model="claude-sonnet-4-5", api_key="sk-not-a-real-key")
    payload = llm._get_request_payload(BAD)
    ids = []
    for m in payload["messages"]:
        cc = m["content"]
        if isinstance(cc, list):
            for x in cc:
                if x.get("type") in ("tool_use", "tool_result"):
                    ids.append((x["type"], x.get("id") or x.get("tool_use_id")))
    print(f"  ChatAnthropic._get_request_payload    -> ACCEPTED; tool blocks {ids}")
    uses = {i for t, i in ids if t == "tool_use"}
    res = {i for t, i in ids if t == "tool_result"}
    print(f"      tool_result ids with NO tool_use in the payload: {sorted(res - uses)}")
    print(f"      tool_use ids with NO tool_result in the payload: {sorted(uses - res)}")
except BaseException as e:  # noqa: BLE001
    print(f"  ChatAnthropic._get_request_payload    -> {type(e).__name__}: {str(e)[:120]}")

try:
    oai = convert_to_openai_messages(BAD)
    print(f"  convert_to_openai_messages           -> ACCEPTED, {len(oai)} messages")
    print(f"      {[(m.get('role'), m.get('tool_call_id') or [t['id'] for t in m.get('tool_calls', [])] or None) for m in oai]}")
except BaseException as e:  # noqa: BLE001
    print(f"  convert_to_openai_messages           -> {type(e).__name__}: {str(e)[:120]}")

try:
    import anthropic
    from anthropic.types import MessageParam
    print(f"  anthropic {anthropic.__version__}: MessageParam is a TypedDict? "
          f"{type(MessageParam).__name__ == '_TypedDictMeta' or hasattr(MessageParam, '__annotations__')}"
          f"  (TypedDicts do no runtime validation)")
except BaseException as e:  # noqa: BLE001
    print(f"  anthropic import -> {type(e).__name__}: {e}")
