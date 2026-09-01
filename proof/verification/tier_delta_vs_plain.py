"""Attack 4: is the corruption DeltaChannel-specific or general to LangGraph?

Same two-process SIGSTOP scenario as proof.py, but the graph is a switch:
  --graph deep   -> deepagents create_deep_agent   (messages = DeltaChannel)
  --graph plain  -> hand-rolled StateGraph on MessagesState / add_messages

Worker mode:  tier_delta_vs_plain.py worker <json-cfg>
Driver mode:  tier_delta_vs_plain.py drive <deep|plain>

Provenance: copied verbatim (module docstring aside) from the adversarial-review
scratchpad as `narrow/nw.py`. This is the evidence behind the Tier 1 / Tier 2
split in ../README.md, docs/reference/langgraph-facts.md and
docs/open-issues/01-checkpointer-fencing.md -- see ../README.md in this
directory for the full evidence-to-claim mapping. Only the sys.path setup below
and the self-re-exec filename were changed, to point at this repo instead of
the scratchpad's throwaway `pristine/` copy of fence.py.
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time

ME = pathlib.Path(__file__).resolve()
HERE = ME.parent
REPO = HERE.parent.parent  # proof/verification -> proof -> repo root
sys.path.insert(0, str(REPO / "spike"))
sys.path.insert(0, str(REPO / "proof"))

import psycopg  # noqa: E402
from _harness import ScriptedChatModel, pg_conninfo  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402

from fence import orphans, setup_database  # noqa: E402  (the proof's own validator)

PY = str(REPO / "spike" / ".venv" / "bin" / "python")

SCRIPT_A = [{"content": "looking up k1", "tool": {"id": "tc_A1", "key": "k1"}},
            {"content": "worker A turn 2", "tool": {"id": "tc_A2", "key": "k2"}},
            {"content": "final answer from A", "tool": None}]
SCRIPT_B = [{"content": "worker B turn 2", "tool": {"id": "tc_B2", "key": "k2"}},
            {"content": "final answer from B", "tool": None}]


@tool
def lookup(key: str) -> str:
    """Look up a key in the knowledge base."""
    return f"value-of-{key}"


class FreezableModel(ScriptedChatModel):
    block_at: int = -1
    ready_file: str = ""
    thaw_file: str = ""

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if self.i == self.block_at:
            pathlib.Path(self.ready_file).write_text(str(os.getpid()))
            while not pathlib.Path(self.thaw_file).exists():
                time.sleep(0.02)
        return super()._generate(messages, stop, run_manager, **kw)


def to_ai(spec: dict) -> AIMessage:
    t = spec.get("tool")
    calls = [{"name": "lookup", "args": {"key": t["key"]}, "id": t["id"]}] if t else []
    return AIMessage(content=spec["content"], tool_calls=calls)


def build_plain(model, saver):
    """A plain react loop on MessagesState: NO DeltaChannel anywhere."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    def call_tools(state):
        last = state["messages"][-1]
        return {"messages": [
            ToolMessage(content=lookup.invoke(tc["args"]), name=tc["name"],
                        tool_call_id=tc["id"]) for tc in last.tool_calls]}

    def route(state):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_node("tools", call_tools)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", route, {"tools": "tools", END: END})
    g.add_edge("tools", "model")
    return g.compile(checkpointer=saver)


def build_deep(model, saver):
    from deepagents import create_deep_agent
    return create_deep_agent(model=model, tools=[lookup], checkpointer=saver)


def build(kind: str, model, saver):
    return (build_plain if kind == "plain" else build_deep)(model, saver)


# ------------------------------------------------------------------ worker ---
def worker() -> int:
    import traceback
    cfg = json.loads(sys.argv[2])
    out: dict = {"role": cfg["role"], "pid": os.getpid(), "ok": False}
    try:
        model = FreezableModel(
            responses=[to_ai(s) for s in cfg["script"]],
            block_at=cfg.get("block_at", -1),
            ready_file=cfg.get("ready_file", ""),
            thaw_file=cfg.get("thaw_file", ""))
        conn = psycopg.connect(cfg["conninfo"], autocommit=True)
        agent = build(cfg["graph"], model, PostgresSaver(conn))
        rc = {"configurable": {"thread_id": cfg["thread_id"]}}
        inp = None if cfg["role"] == "B" else {"messages": [HumanMessage("go")]}
        res = agent.invoke(inp, rc, durability="sync")
        out["ok"] = True
        out["messages"] = [[type(m).__name__, str(m.content)[:40],
                            [c["id"] for c in (getattr(m, "tool_calls", None) or [])],
                            getattr(m, "tool_call_id", None)] for m in res["messages"]]
    except BaseException as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}".strip()[:300]
        out["tb"] = traceback.format_exc().strip().splitlines()[-1]
    pathlib.Path(cfg["out_file"]).write_text(json.dumps(out))
    return 0 if out["ok"] else 1


# ------------------------------------------------------------------ driver ---
def chain(conninfo, tid):
    with psycopg.connect(conninfo, autocommit=True,
                         row_factory=psycopg.rows.dict_row) as c:
        return c.execute(
            "select checkpoint_id id, parent_checkpoint_id parent,"
            " metadata->>'step' step from checkpoints where thread_id=%s"
            " and checkpoint_ns='' order by checkpoint_id", (tid,)).fetchall()


def forks(rows):
    kids: dict = {}
    for r in rows:
        if r["parent"]:
            kids.setdefault(r["parent"], []).append(r["id"])
    return [(p, v) for p, v in kids.items() if len(v) > 1]


def writes_at(conninfo, tid, cid, channel="messages"):
    with psycopg.connect(conninfo, autocommit=True) as c:
        rows = c.execute(
            "select task_id, idx, type, blob from checkpoint_writes where"
            " thread_id=%s and checkpoint_ns='' and checkpoint_id=%s and channel=%s"
            " order by task_id, idx", (tid, cid, channel)).fetchall()
    serde = JsonPlusSerializer()
    return [(t, i, serde.loads_typed((ty, b))) for t, i, ty, b in rows]


def blobs_for(conninfo, tid, channel="messages"):
    with psycopg.connect(conninfo, autocommit=True) as c:
        return c.execute(
            "select channel, version, type, length(blob) from checkpoint_blobs"
            " where thread_id=%s and channel=%s order by version",
            (tid, channel)).fetchall()


def drive(kind: str) -> int:
    conninfo = pg_conninfo(f"da_narrow_{kind}")
    setup_database(conninfo)
    tid = f"narrow-{kind}"
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nw-"))
    common = {"conninfo": conninfo, "thread_id": tid, "graph": kind}
    a = dict(common, role="A", script=SCRIPT_A, block_at=1,
             ready_file=str(tmp / "ready"), thaw_file=str(tmp / "thaw"),
             out_file=str(tmp / "a.json"))
    b = dict(common, role="B", script=SCRIPT_B, block_at=-1,
             out_file=str(tmp / "b.json"))

    print(f"\n{'=' * 74}\nGRAPH = {kind}\n{'=' * 74}")
    # what channel does `messages` actually use?
    dummy = build(kind, ScriptedChatModel(responses=[]), None)
    ch = dummy.channels["messages"]
    print(f"  messages channel: {type(ch).__module__}.{type(ch).__name__}")

    pa = subprocess.Popen([PY, str(ME), "worker", json.dumps(a)],
                          stderr=open(str(tmp / "a.err"), "w"))
    end = time.time() + 90
    while not (tmp / "ready").exists() and time.time() < end:
        time.sleep(0.02)
    if not (tmp / "ready").exists():
        pa.kill()
        raise SystemExit("A never reached its 2nd model call")
    pid = int((tmp / "ready").read_text())
    os.kill(pid, signal.SIGSTOP)
    st = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                        capture_output=True, text=True).stdout.strip()
    snap_frozen = chain(conninfo, tid)
    parent = snap_frozen[-1]["id"]
    print(f"  A pid {pid} SIGSTOP'd (ps state={st!r}); chain at freeze:"
          f" {len(snap_frozen)} ckpts, tip step {snap_frozen[-1]['step']}"
          f" ...{parent[-8:]}")

    pb = subprocess.Popen([PY, str(ME), "worker", json.dumps(b)],
                          stderr=open(str(tmp / "b.err"), "w"))
    pb.wait(timeout=180)
    b_out = json.loads((tmp / "b.json").read_text())
    print(f"  B invoke(None, config): ok={b_out['ok']} err={b_out.get('error')}")
    snap_after_b = chain(conninfo, tid)
    b_writes = writes_at(conninfo, tid, parent)
    print(f"  writes at fork parent AFTER B, BEFORE A thaws: {len(b_writes)} rows"
          f" task_ids={[t[:8] for t, _, _ in b_writes]}")

    (tmp / "thaw").write_text("go")
    os.kill(pid, signal.SIGCONT)
    try:
        pa.wait(timeout=180)
    except subprocess.TimeoutExpired:
        pa.kill()
        print("  !! A deadlocked")
    a_out = json.loads((tmp / "a.json").read_text())
    print(f"  A after thaw: ok={a_out['ok']} err={a_out.get('error')}")
    for who, f in (("A", tmp / "a.err"), ("B", tmp / "b.err")):
        t = f.read_text().strip()
        for line in t.splitlines()[-4:]:
            print(f"    [{who} stderr] {line}")

    snap_final = chain(conninfo, tid)
    fk = forks(snap_final)
    print(f"\n  FORK: {len(fk)} forked parent(s)"
          + (f"  parent ...{fk[0][0][-8:]} -> "
             + ", ".join('...' + c[-8:] for c in fk[0][1]) if fk else ""))
    print(f"  fork parent == freeze-time tip: {bool(fk) and fk[0][0] == parent}")

    stored = writes_at(conninfo, tid, parent)
    ids = sorted({c["id"] for _, _, v in stored
                  for m in (v if isinstance(v, list) else [v])
                  for c in (getattr(m, "tool_calls", None) or [])})
    print(f"  messages write rows at fork parent AFTER A thawed: {len(stored)}"
          f" task_ids={[t[:8] for t, _, _ in stored]} tool_calls={ids}")
    print(f"  checkpoint_blobs rows for channel 'messages': {blobs_for(conninfo, tid)}")

    conn = psycopg.connect(conninfo, autocommit=True)
    agent = build(kind, ScriptedChatModel(responses=[]), PostgresSaver(conn))
    stt = agent.get_state({"configurable": {"thread_id": tid}})
    msgs = stt.values.get("messages", [])
    print("  thread as read back from the tip:")
    for m in msgs:
        print(f"    {type(m).__name__:<14} {str(m.content)[:26]:<28}"
              f" tool_calls={[c['id'] for c in (getattr(m, 'tool_calls', None) or [])] or '-'}"
              f" tool_call_id={getattr(m, 'tool_call_id', None) or '-'}")
    probs = orphans(msgs)
    print(f"  orphans() -> {probs or 'NONE (well-formed)'}")
    by = {r["id"]: r for r in snap_final}
    reach, cur = set(), snap_final[-1]
    while cur:
        reach.add(cur["id"])
        cur = by.get(cur["parent"])
    lost = [r["id"] for r in snap_after_b if r["id"] not in reach]
    print(f"  B checkpoints unreachable from tip: {len(lost)}")
    print(f"  .next at tip = {stt.next}")
    print(f"  B returned to its caller: {b_out.get('messages', [[None,'?']])[-1][1]!r}")
    print(f"  tip says: {str(msgs[-1].content)[:40]!r}" if msgs else "  tip: EMPTY")
    print(f"\n  SUMMARY[{kind}]: forked={bool(fk)}  malformed={bool(probs)}"
          f"  orphan_tool_result="
          f"{any('has no matching' in p for p in probs)}  lost_B_ckpts={len(lost)}")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        sys.exit(worker())
    sys.exit(drive(sys.argv[2]))
