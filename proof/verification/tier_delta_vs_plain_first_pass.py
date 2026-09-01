"""E1 -- verify CORRECTION 1: run the identical freeze/thaw scenario against
deepagents (DeltaChannel messages) and a plain StateGraph on MessagesState
(BinaryOperatorAggregate via add_messages).  Measure the review's table.

usage: tier_delta_vs_plain_first_pass.py               # orchestrator, both flavors
       tier_delta_vs_plain_first_pass.py <json cfg>    # worker (spawned)

Provenance: copied VERBATIM (not one character changed) from the adversarial-
review scratchpad as `e1_plain_vs_delta.py` -- this is the first-pass version
of the plain-vs-delta comparison; `tier_delta_vs_plain.py` is the narrower
rewrite. Both back the Tier 1 / Tier 2 split in ../README.md,
docs/reference/langgraph-facts.md and docs/open-issues/01-checkpointer-fencing.md.
No path fix was needed: this script already hardcodes this repo's absolute
path. See ../README.md for the full evidence-to-claim mapping.
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
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spike"))
sys.path.insert(0, str(ROOT / "proof"))

import psycopg  # noqa: E402
from _harness import ScriptedChatModel, pg_conninfo  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402

from fence import orphans  # noqa: E402

PY = str(ROOT / "spike" / ".venv" / "bin" / "python")
ME = str(pathlib.Path(__file__).resolve())

SCRIPT_A = [{"content": "looking up k1", "tool": {"id": "tc_A1", "key": "k1"}},
            {"content": "worker A turn 2", "tool": {"id": "tc_A2", "key": "k2"}},
            {"content": "final answer from A", "tool": None}]
SCRIPT_B = [{"content": "worker B turn 2", "tool": {"id": "tc_B2", "key": "k2"}},
            {"content": "final answer from B", "tool": None}]


@tool
def lookup(key: str) -> str:
    """Look up a key in the knowledge base."""
    return f"value-of-{key}"


def to_ai(spec):
    t = spec.get("tool")
    calls = [{"name": "lookup", "args": {"key": t["key"]}, "id": t["id"]}] if t else []
    return AIMessage(content=spec["content"], tool_calls=calls)


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


def plain_graph(model, saver):
    """A hand-rolled react loop on MessagesState: `messages` is a plain
    BinaryOperatorAggregate over add_messages, so a full-list blob is written."""
    def model_node(state):
        return {"messages": [model.invoke(state["messages"])]}

    def tool_node(state):
        last = state["messages"][-1]
        return {"messages": [
            ToolMessage(content=lookup.invoke(tc["args"]), tool_call_id=tc["id"],
                        name="lookup") for tc in last.tool_calls]}

    def route(state):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    g = StateGraph(MessagesState)
    g.add_node("model", model_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", route, {"tools": "tools", END: END})
    g.add_edge("tools", "model")
    return g.compile(checkpointer=saver)


def build(flavor, model, saver):
    if flavor == "delta":
        from deepagents import create_deep_agent
        return create_deep_agent(model=model, tools=[lookup], checkpointer=saver)
    return plain_graph(model, saver)


# ------------------------------------------------------------------- worker ---
def worker(cfg) -> int:
    out = {"role": cfg["role"], "pid": os.getpid(), "ok": False}
    try:
        model = FreezableModel(responses=[to_ai(s) for s in cfg["script"]],
                               block_at=cfg.get("block_at", -1),
                               ready_file=cfg.get("ready_file", ""),
                               thaw_file=cfg.get("thaw_file", ""))
        conn = psycopg.connect(cfg["conninfo"], autocommit=True)
        saver = PostgresSaver(conn)
        agent = build(cfg["flavor"], model, saver)
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


# ------------------------------------------------------------- orchestrator ---
def chain(conninfo, tid):
    with psycopg.connect(conninfo, autocommit=True,
                         row_factory=psycopg.rows.dict_row) as c:
        return c.execute(
            "select checkpoint_id id, parent_checkpoint_id parent,"
            " metadata->>'step' step from checkpoints where thread_id=%s"
            " and checkpoint_ns='' order by checkpoint_id", (tid,)).fetchall()


def forks(rows):
    kids = {}
    for r in rows:
        if r["parent"]:
            kids.setdefault(r["parent"], []).append(r["id"])
    return [(p, v) for p, v in kids.items() if len(v) > 1]


def run(flavor, conninfo):
    tid = f"e1-{flavor}"
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"e1-{flavor}-"))
    common = {"conninfo": conninfo, "thread_id": tid, "flavor": flavor}
    a = dict(common, role="A", script=SCRIPT_A, block_at=1,
             ready_file=str(tmp / "ready"), thaw_file=str(tmp / "thaw"),
             out_file=str(tmp / "a.json"))
    b = dict(common, role="B", script=SCRIPT_B, block_at=-1,
             out_file=str(tmp / "b.json"))

    pa = subprocess.Popen([PY, ME, json.dumps(a)],
                          stderr=open(str(tmp / "a.err"), "w"))
    end = time.time() + 90
    while not (tmp / "ready").exists() and time.time() < end:
        time.sleep(0.02)
    if not (tmp / "ready").exists():
        pa.kill()
        raise SystemExit(f"[{flavor}] A never reached 2nd model call:"
                         f" {pathlib.Path(str(tmp / 'a.err')).read_text()[-2000:]}")
    pid = int((tmp / "ready").read_text())
    os.kill(pid, signal.SIGSTOP)
    st = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                        capture_output=True, text=True).stdout.strip()
    snap_frozen = chain(conninfo, tid)
    fork_parent = snap_frozen[-1]["id"]

    pb = subprocess.Popen([PY, ME, json.dumps(b)],
                          stderr=open(str(tmp / "b.err"), "w"))
    pb.wait(timeout=240)
    b_out = json.loads((tmp / "b.json").read_text())
    snap_after_b = chain(conninfo, tid)

    (tmp / "thaw").write_text("go")
    os.kill(pid, signal.SIGCONT)
    try:
        pa.wait(timeout=240)
    except subprocess.TimeoutExpired:
        pa.kill()
    a_out = json.loads((tmp / "a.json").read_text())
    snap_final = chain(conninfo, tid)

    # -- measurements ------------------------------------------------------
    serde = JsonPlusSerializer()
    with psycopg.connect(conninfo, autocommit=True) as c:
        rows = c.execute(
            "select task_id, type, blob from checkpoint_writes where thread_id=%s"
            " and checkpoint_ns='' and checkpoint_id=%s and channel='messages'",
            (tid, fork_parent)).fetchall()
        nblobs = c.execute(
            "select count(*) from checkpoint_blobs where thread_id=%s and channel='messages'",
            (tid,)).fetchone()[0]
        blob_ch = c.execute(
            "select channel, count(*) from checkpoint_blobs where thread_id=%s"
            " group by channel order by channel", (tid,)).fetchall()
        allwrites = c.execute(
            "select checkpoint_id, task_id, idx, channel, type, blob from checkpoint_writes"
            " where thread_id=%s and checkpoint_ns='' order by checkpoint_id, task_id, idx",
            (tid,)).fetchall()
    stored = [(t, serde.loads_typed((ty, b))) for t, ty, b in rows]
    stored_ids = sorted({c["id"] for _, v in stored
                         for m in (v if isinstance(v, list) else [v])
                         for c in (getattr(m, "tool_calls", None) or [])})

    by_id, reach, cur = {r["id"]: r for r in snap_final}, set(), snap_final[-1]
    while cur:
        reach.add(cur["id"])
        cur = by_id.get(cur["parent"])
    lost = [r["id"] for r in snap_after_b if r["id"] not in reach]

    # read the tip back
    conn = psycopg.connect(conninfo, autocommit=True)
    tipgraph = build(flavor, ScriptedChatModel(responses=[]), PostgresSaver(conn))
    st_snap = tipgraph.get_state({"configurable": {"thread_id": tid}})
    msgs = st_snap.values.get("messages", [])
    probs = orphans(msgs)

    # where is every tool_use / tool_result id durably stored?
    where = {}
    for cid, task, idx, ch, ty, b in allwrites:
        try:
            v = serde.loads_typed((ty, b))
        except Exception:
            continue
        for m in (v if isinstance(v, list) else [v]):
            for tc in (getattr(m, "tool_calls", None) or []):
                where.setdefault("tool_use:" + tc["id"], []).append(
                    f"write@...{cid[-8:]}/{task[:8]}/{ch}")
            t = getattr(m, "tool_call_id", None)
            if t:
                where.setdefault("tool_result:" + t, []).append(
                    f"write@...{cid[-8:]}/{task[:8]}/{ch}")
    # and in blobs
    with psycopg.connect(conninfo, autocommit=True) as c:
        brows = c.execute(
            "select checkpoint_id_placeholder from (select 1) t limit 0") if False else None
        brows = c.execute(
            "select channel, version, type, blob from checkpoint_blobs where thread_id=%s",
            (tid,)).fetchall()
    for ch, ver, ty, b in brows:
        if ch != "messages" or b is None:
            continue
        try:
            v = serde.loads_typed((ty, bytes(b)))
        except Exception:
            continue
        for m in (v if isinstance(v, list) else [v]):
            for tc in (getattr(m, "tool_calls", None) or []):
                where.setdefault("tool_use:" + tc["id"], []).append(f"blob@{ver}")
            t = getattr(m, "tool_call_id", None)
            if t:
                where.setdefault("tool_result:" + t, []).append(f"blob@{ver}")

    return {
        "flavor": flavor, "ps": st, "fork_parent": fork_parent,
        "n_frozen": len(snap_frozen), "n_after_b": len(snap_after_b),
        "n_final": len(snap_final),
        "forks": [(p[-8:], [x[-8:] for x in v]) for p, v in forks(snap_final)],
        "fork_parent_is_freeze_tip": bool(forks(snap_final))
                                     and forks(snap_final)[0][0] == fork_parent,
        "rows_at_fork_parent": len(stored),
        "task_ids_at_fork_parent": [t[:8] for t, _ in stored],
        "stored_tool_calls_at_fork_parent": stored_ids,
        "b_unreachable": len(lost),
        "a_ok": a_out["ok"], "b_ok": b_out["ok"],
        "a_err": a_out.get("error"), "b_err": b_out.get("error"),
        "b_returned": b_out.get("messages", [[None, None]])[-1][1] if b_out["ok"] else None,
        "tip_says": str(msgs[-1].content)[:40] if msgs else None,
        "blobs_messages": nblobs,
        "blobs_by_channel": blob_ch,
        "tip_list": [[type(m).__name__, str(m.content)[:24],
                      [c["id"] for c in (getattr(m, "tool_calls", None) or [])],
                      getattr(m, "tool_call_id", None)] for m in msgs],
        "orphans": probs,
        "id_locations": where,
    }


def setup(conninfo):
    with psycopg.connect(conninfo, autocommit=True) as c:
        for t in ("checkpoints", "checkpoint_blobs", "checkpoint_writes",
                  "checkpoint_migrations"):
            c.execute(f"drop table if exists {t} cascade")
        PostgresSaver(c).setup()


def main():
    for flavor in ("delta", "plain"):
        conninfo = pg_conninfo(f"e1_{flavor}")
        setup(conninfo)
        r = run(flavor, conninfo)
        print("=" * 78)
        print(f"FLAVOR: {flavor}   ({'deepagents DeltaChannel' if flavor == 'delta' else 'plain MessagesState / add_messages'})")
        print("=" * 78)
        for k in ("ps", "n_frozen", "n_after_b", "n_final", "forks",
                  "fork_parent_is_freeze_tip", "rows_at_fork_parent",
                  "task_ids_at_fork_parent", "stored_tool_calls_at_fork_parent",
                  "b_unreachable", "a_ok", "b_ok", "a_err", "b_err",
                  "b_returned", "tip_says", "blobs_messages", "blobs_by_channel"):
            print(f"  {k:34} = {r[k]}")
        print("  tip_list:")
        for row in r["tip_list"]:
            print(f"    {row}")
        print(f"  orphans ({len(r['orphans'])}):")
        for p in r["orphans"]:
            print(f"    - {p}")
        print("  where every id is durably stored:")
        for k in sorted(r["id_locations"]):
            print(f"    {k:22} {r['id_locations'][k]}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(worker(json.loads(sys.argv[1])))
    main()
