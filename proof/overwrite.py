"""The OVERWRITE proof. The other ON CONFLICT clause, end to end.

    spike/.venv/bin/python proof/overwrite.py     # ~25s, exit 0 on PROVEN

Companion to proof.py, which shows the DROP: `put_writes` on a batch containing
any ordinary channel uses `INSERT .. ON CONFLICT DO NOTHING`, so a stale
worker's `messages` write is silently discarded. This file shows the strictly
worse sibling: a batch whose channels are ALL in `WRITES_IDX_MAP` uses
`UPSERT .. ON CONFLICT DO UPDATE`, so the stale worker does not lose -- it
REPLACES the live worker's row. Those channels carry control data
(`__resume__`, `__interrupt__`, `__error__`, `__scheduled__`), not transcript.

Three levels of evidence, labelled everywhere:
  READ IN SOURCE  -- quoted from the installed langgraph, with file:line
  HAND-DRIVEN SQL -- statements issued by this file against real Postgres
  REAL GRAPH RUN  -- observed through graph.invoke() in separate OS processes

Part 1  hand-driven: which statement replaces, which drops, and the idx map.
Part 1b real graph run: a write tap enumerating which batches a genuine
        interrupt/resume cycle actually routes to the DO UPDATE statement.
Part 2  real graph run: two OS processes on one thread_id. The frozen worker's
        `__resume__` overwrites the survivor's, and a later worker then resumes
        the run on a human input that was never given to it.
Part 3  the same Part 2 with proof/fence.py switched on.
"""
from __future__ import annotations

import json
import operator
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
from typing import Annotated, TypedDict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "spike"))

import psycopg  # noqa: E402
from _harness import pg_conninfo  # noqa: E402
from langgraph._internal._constants import NULL_TASK_ID  # noqa: E402  (private)
from langgraph.checkpoint.base import WRITES_IDX_MAP  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from langgraph.checkpoint.postgres.base import (  # noqa: E402
    INSERT_CHECKPOINT_WRITES_SQL, UPSERT_CHECKPOINT_WRITES_SQL)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from langgraph.constants import START  # noqa: E402
from langgraph.graph import StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

from fence import (FencedPostgresSaver, claim, install_fence, read_fence,  # noqa: E402
                   setup_database)

PY = str(HERE.parents[0] / "spike" / ".venv" / "bin" / "python")
SERDE = JsonPlusSerializer()
LOG = None
FAILURES: list[str] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    if LOG is not None:
        LOG.write(msg + "\n")


def check(ok: bool, label: str, detail: str = "") -> bool:
    say(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)
    return ok


# --------------------------------------------------------------- the graph ---
# A plain StateGraph, not deepagents: `__resume__` is a LangGraph control
# channel, so nothing here depends on how the transcript channel is declared.
# TWO interrupt() calls in one node, because that is what makes the durable
# `(task_id, __resume__)` row -- the accumulated list of already-consumed human
# inputs -- outlive the run that produced it, and therefore observable by the
# NEXT worker. See langgraph/types.py:851-974 and pregel/_runner.py:585-591.
class S(TypedDict):
    log: Annotated[list[str], operator.add]


def approve(state: S) -> dict:
    first = interrupt("approve step 1?")
    second = interrupt("approve step 2?")
    return {"log": [f"{first}|{second}"]}


def build(saver):
    g = StateGraph(S)
    g.add_node("approve", approve)
    g.add_edge(START, "approve")
    return g.compile(checkpointer=saver)


# ---------------------------------------------------------------- the gate ---
class _Gate:
    """Stall every write of thread state until a file appears.

    Guards exactly the fence's seam -- `_cursor(pipeline=True)`, i.e. put,
    put_writes, delete_thread -- so a gated worker can still READ its
    checkpoint and run its node, but issues no SQL that mutates the thread.
    That is what lets the orchestrator order two workers deterministically:
    worker A loads checkpoint C, is frozen holding an unissued `__resume__`
    write for C, and issues it only after worker B has finished.

    The gate makes the stall deterministic; the SIGSTOP that follows makes it a
    real freeze rather than a cooperative pause.
    """

    gate_ready: str = ""
    gate_thaw: str = ""

    def _cursor(self, *, pipeline: bool = False):
        if pipeline and self.gate_thaw:
            pathlib.Path(self.gate_ready).write_text(str(os.getpid()))
            while not pathlib.Path(self.gate_thaw).exists():
                time.sleep(0.02)
        return super()._cursor(pipeline=pipeline)


class GatedSaver(_Gate, PostgresSaver):
    pass


class GatedFencedSaver(_Gate, FencedPostgresSaver):
    pass


# ------------------------------------------------------------------ probes ---
def chain(conninfo: str, tid: str) -> list[dict]:
    with psycopg.connect(conninfo, autocommit=True,
                         row_factory=psycopg.rows.dict_row) as c:
        return c.execute(
            "select checkpoint_id id, parent_checkpoint_id parent,"
            " metadata->>'step' step from checkpoints"
            " where thread_id=%s and checkpoint_ns='' order by checkpoint_id",
            (tid,)).fetchall()


def forks(rows: list[dict]) -> list[tuple[str, list[str]]]:
    kids: dict[str, list[str]] = {}
    for r in rows:
        if r["parent"]:
            kids.setdefault(r["parent"], []).append(r["id"])
    return [(p, v) for p, v in kids.items() if len(v) > 1]


def writes_at(conninfo: str, tid: str, cid: str) -> list[dict]:
    """Every checkpoint_writes row at one checkpoint, value deserialized."""
    with psycopg.connect(conninfo, autocommit=True) as c:
        rows = c.execute(
            "select task_id, task_path, idx, channel, type, blob from"
            " checkpoint_writes where thread_id=%s and checkpoint_ns=''"
            " and checkpoint_id=%s order by task_id, idx", (tid, cid)).fetchall()
    out = []
    for task_id, task_path, idx, channel, t, b in rows:
        try:
            v = SERDE.loads_typed((t, bytes(b)))
        except Exception:  # noqa: BLE001
            v = "<undeserializable>"
        out.append({"task_id": task_id, "task_path": task_path, "idx": idx,
                    "channel": channel, "value": v})
    return out


def ctl(rows: list[dict], task_id: str, idx: int):
    """The value of one control row, or MISSING."""
    for r in rows:
        if r["task_id"] == task_id and r["idx"] == idx:
            return r["value"]
    return "<MISSING>"


def show_ctl(rows: list[dict], task: str, header: str) -> None:
    say(header)
    for r in rows:
        who = "NULL_TASK_ID" if r["task_id"] == NULL_TASK_ID else \
            ("task " + r["task_id"][:8] if r["task_id"] == task else r["task_id"][:8])
        say(f"      {who:<14} idx={r['idx']:<3} {r['channel']:<14}"
            f" value={str(r['value'])[:60]}")


# ------------------------------------------------------ PART 1: raw the SQL ---
def part1(conninfo: str) -> None:
    say("\n" + "=" * 78)
    say("PART 1 -- HAND-DRIVEN SQL: which clause replaces, which one drops")
    say("=" * 78)
    say("  READ IN SOURCE  langgraph/checkpoint/postgres/__init__.py:363-367")
    say("    query = (self.UPSERT_CHECKPOINT_WRITES_SQL       # DO UPDATE")
    say("             if all(w[0] in WRITES_IDX_MAP for w in writes)")
    say("             else self.INSERT_CHECKPOINT_WRITES_SQL) # DO NOTHING")
    say("  READ IN SOURCE  langgraph/checkpoint/base/__init__.py:795")
    say(f"    WRITES_IDX_MAP = {WRITES_IDX_MAP}")
    say("  READ IN SOURCE  postgres/base.py:146-159 -- both statements conflict on")
    say("    (thread_id, checkpoint_ns, checkpoint_id, task_id, idx); DO UPDATE SET")
    say("    lists channel, type, blob -- and NOT task_path.")

    say("\n  HAND-DRIVEN SQL  two writers, one conflict key, both real statements:")
    got = {}
    with psycopg.connect(conninfo, autocommit=True) as c:
        for label, sql in (("INSERT .. ON CONFLICT DO NOTHING", INSERT_CHECKPOINT_WRITES_SQL),
                           ("UPSERT .. ON CONFLICT DO UPDATE ", UPSERT_CHECKPOINT_WRITES_SQL)):
            t = "p1-" + ("nothing" if "DO NOTHING" in sql else "update")
            c.execute("delete from checkpoint_writes where thread_id=%s", (t,))
            for who in ("survivor", "zombie"):
                c.execute(sql, (t, "", "cp", "task", f"path-of-{who}", -4,
                                "__resume__", "text", who.upper().encode()))
            rows = c.execute(
                "select count(*), min(task_path), min(convert_from(blob,'utf8'))"
                " from checkpoint_writes where thread_id=%s", (t,)).fetchone()
            c.execute("delete from checkpoint_writes where thread_id=%s", (t,))
            got[label] = rows[2]
            say(f"    {label} -> rows={rows[0]}  blob={rows[2]!r}"
                f"  task_path={rows[1]!r}")
    check(got["INSERT .. ON CONFLICT DO NOTHING"] == "SURVIVOR"
          and got["UPSERT .. ON CONFLICT DO UPDATE "] == "ZOMBIE",
          "at one conflict key DO NOTHING keeps the FIRST writer's blob and"
          " DO UPDATE keeps the SECOND's",
          "so the loser of a DO UPDATE collision is the live worker, not the stale one")
    say("    note: task_path is not in the SET list, so after a DO UPDATE the row"
        " carries\n          the SECOND writer's blob under the FIRST writer's"
        " task_path. Since idx is\n          derived from the channel for special"
        " writes, the reachable effect of the SET\n          list is blob+type"
        " replacement at a fixed (channel, idx) -- not a channel swap.")

    say("\n  HAND-DRIVEN SQL  the same collision through the real put_writes() API,")
    say("  two savers on two connections, so the branch is chosen by langgraph:")
    cfg = {"configurable": {"thread_id": "", "checkpoint_ns": "", "checkpoint_id": "cp"}}
    outcomes = {}
    with psycopg.connect(conninfo, autocommit=True) as ca, \
            psycopg.connect(conninfo, autocommit=True) as cb, \
            psycopg.connect(conninfo, autocommit=True) as cr:
        sa, sb = PostgresSaver(ca), PostgresSaver(cb)
        for label, batch in (
            ("ALL channels special  [('__resume__', v)]",
             lambda v: [("__resume__", v)]),
            ("MIXED  [('__resume__', v), ('log', [v])]",
             lambda v: [("__resume__", v), ("log", [v])]),
        ):
            t = "p1-api-" + ("pure" if "ALL" in label else "mixed")
            cr.execute("delete from checkpoint_writes where thread_id=%s", (t,))
            c2 = dict(cfg); c2["configurable"] = dict(cfg["configurable"], thread_id=t)
            sa.put_writes(c2, batch("SURVIVOR"), "task")
            sb.put_writes(c2, batch("ZOMBIE"), "task")
            rows = cr.execute(
                "select idx, channel, type, blob from checkpoint_writes"
                " where thread_id=%s order by idx", (t,)).fetchall()
            vals = {ch: SERDE.loads_typed((ty, bytes(bl))) for _, ch, ty, bl in rows}
            outcomes[label] = vals.get("__resume__")
            cr.execute("delete from checkpoint_writes where thread_id=%s", (t,))
            say(f"    {label:<42} -> idx={[r[0] for r in rows]}"
                f" __resume__={vals.get('__resume__')!r}")
    check(outcomes["ALL channels special  [('__resume__', v)]"] == "ZOMBIE"
          and outcomes["MIXED  [('__resume__', v), ('log', [v])]"] == "SURVIVOR",
          "put_writes ITSELF routes a pure-special batch to DO UPDATE (second writer"
          " wins) and a mixed batch to DO NOTHING (first writer wins)",
          "the branch selection is measured, not inferred from reading the source")

    say("\n  HAND-DRIVEN SQL  the idx a channel lands on (the conflict key's last"
        " column):")
    with psycopg.connect(conninfo, autocommit=True) as c:
        s = PostgresSaver(c)
        c2 = {"configurable": {"thread_id": "p1-idx", "checkpoint_ns": "",
                               "checkpoint_id": "cp"}}
        c.execute("delete from checkpoint_writes where thread_id='p1-idx'")
        s.put_writes(c2, [(ch, "v") for ch in WRITES_IDX_MAP], "t-special")
        s.put_writes(c2, [("log", "a"), ("other", "b")], "t-ordinary")
        rows = c.execute("select task_id, idx, channel from checkpoint_writes"
                         " where thread_id='p1-idx' order by task_id, idx").fetchall()
        c.execute("delete from checkpoint_writes where thread_id='p1-idx'")
    for tid, idx, ch in rows:
        say(f"    {tid:<12} idx={idx:<3} {ch}")
    special = {ch: idx for tid, idx, ch in rows if tid == "t-special"}
    check(special == WRITES_IDX_MAP,
          "each special channel lands on its OWN fixed negative idx, so two workers"
          " writing the same control channel at one checkpoint always collide",
          f"{special}")


# ------------------------------------ PART 1b: which batches a real run makes ---
class TapSaver(PostgresSaver):
    """Records every put_writes batch and which statement it selects."""

    def put_writes(self, config, writes, task_id, task_path=""):
        pure = all(w[0] in WRITES_IDX_MAP for w in writes)
        self.tap.append((config["configurable"]["checkpoint_id"][-8:],
                         "NULL_TASK_ID" if task_id == NULL_TASK_ID else task_id[:8],
                         [w[0] for w in writes],
                         "UPSERT/DO UPDATE" if pure else "INSERT/DO NOTHING"))
        return super().put_writes(config, writes, task_id, task_path)


def part1b(conninfo: str) -> None:
    say("\n" + "=" * 78)
    say("PART 1b -- REAL GRAPH RUN: which batches actually take the DO UPDATE path")
    say("=" * 78)
    tid = "ow-tap"
    with psycopg.connect(conninfo, autocommit=True) as conn:
        saver = TapSaver(conn)
        saver.tap = []
        g = build(saver)
        rc = {"configurable": {"thread_id": tid}}
        say("  one worker, one thread: start -> interrupt -> resume -> interrupt"
            " -> resume -> done")
        g.invoke({"log": ["start"]}, rc, durability="sync")
        g.invoke(Command(resume="R1"), rc, durability="sync")
        st = g.invoke(Command(resume="R2"), rc, durability="sync")
        tap = list(saver.tap)
    say("  every put_writes batch it issued:")
    for cid, task, chans, q in tap:
        say(f"    cp ...{cid}  {task:<12} {str(chans):<46} -> {q}")
    upserts = [t for t in tap if "UPSERT" in t[3]]
    check(bool(upserts),
          "a real graph run DOES route batches to the DO UPDATE statement",
          f"{len(upserts)} of {len(tap)} batches: "
          + "; ".join(str(t[2]) for t in upserts))
    say(f"  final state: log={st.get('log')}")
    say("  READ IN SOURCE, matching the three shapes above:")
    say("    pregel/_loop.py:919-926  Command(resume=v) -> put_writes(NULL_TASK_ID,"
        " [(RESUME, v)])")
    say("    pregel/_runner.py:585-591  GraphInterrupt -> put_writes(task_id,"
        " [(INTERRUPT, ..)] + RESUME)")
    say("    pregel/_runner.py:596-604  an unhandled node error with no prior channel"
        " write\n      -> put_writes(task_id, [(ERROR, exc)]) -- also all-special,"
        " also DO UPDATE (not exercised here)")


# --------------------------------------------------- PART 2/3: the scenario ---
def spawn(cfg: dict) -> subprocess.Popen:
    err = open(cfg["out_file"] + ".err", "w")
    return subprocess.Popen([PY, str(HERE / "overwrite.py"), "--worker",
                             json.dumps(cfg)], stderr=err)


def read_out(cfg: dict, who: str) -> dict:
    p = pathlib.Path(cfg["out_file"])
    if p.exists():
        return json.loads(p.read_text())
    FAILURES.append(f"worker {who} produced no result file (hang or hard crash)")
    say(f"  !! worker {who} wrote no result file ({p})")
    return {"ok": False, "error": "no result file"}


def echo_stderr(cfg: dict, who: str) -> None:
    txt = pathlib.Path(cfg["out_file"] + ".err").read_text().strip()
    for line in txt.splitlines()[-4:]:
        say(f"    [{who} stderr] {line}")


def scenario(conninfo: str, fenced: bool) -> None:
    tid = "ow-on" if fenced else "ow-off"
    say("\n" + "=" * 78)
    say(f"PART {'3' if fenced else '2'} -- REAL GRAPH RUN, TWO OS PROCESSES,"
        f" FENCING {'ON' if fenced else 'OFF'}")
    say("=" * 78)
    if fenced:
        install_fence(conninfo, tid, "A")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="overwrite-"))

    # stage 0 -- the run that got interrupted. In-process, unfenced, real invoke.
    with psycopg.connect(conninfo, autocommit=True) as conn:
        build(PostgresSaver(conn)).invoke({"log": ["start"]},
                                          {"configurable": {"thread_id": tid}},
                                          durability="sync")
    snap0 = chain(conninfo, tid)
    cp = snap0[-1]["id"]
    rows0 = writes_at(conninfo, tid, cp)
    task = next((r["task_id"] for r in rows0 if r["channel"] == "__interrupt__"), "?")
    say(f"  stage 0  a run reached interrupt() and paused. {len(snap0)} checkpoints,"
        f" tip ...{cp[-8:]} step {snap0[-1]['step']}")
    show_ctl(rows0, task, "    control rows at the tip:")

    # stage 1 -- worker A loads that checkpoint, then freezes before writing.
    a_cfg = {"conninfo": conninfo, "thread_id": tid, "role": "A", "resume": "A-VALUE",
             "gate_ready": str(tmp / "ready"), "gate_thaw": str(tmp / "thaw"),
             "out_file": str(tmp / "a.json"), "fence": 1 if fenced else None}
    b_cfg = {"conninfo": conninfo, "thread_id": tid, "role": "B", "resume": "B-VALUE",
             "out_file": str(tmp / "b.json"), "fence": 2 if fenced else None}
    pa = spawn(a_cfg)
    end = time.time() + 60
    while not (tmp / "ready").exists() and time.time() < end:
        time.sleep(0.02)
    if not (tmp / "ready").exists():
        pa.kill()
        FAILURES.append("worker A never reached its first write")
        return
    pid = int((tmp / "ready").read_text())
    os.kill(pid, signal.SIGSTOP)
    state = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                           capture_output=True, text=True).stdout.strip()
    say(f"  stage 1  worker A (pid {pid}) resumed with 'A-VALUE', loaded"
        f" ...{cp[-8:]}, and is SIGSTOP'd\n           holding an unissued"
        f" __resume__ write for it (ps state={state!r}; T = stopped)")

    # stage 2 -- worker B takes the thread over and resumes with a different value.
    pb = spawn(b_cfg)
    try:
        pb.wait(timeout=180)
    except subprocess.TimeoutExpired:
        pb.kill()
        FAILURES.append("worker B hung")
    b_out = read_out(b_cfg, "B")
    rows_b = writes_at(conninfo, tid, cp)
    snap_b = chain(conninfo, tid)
    say(f"  stage 2  worker B resumed the SAME thread with 'B-VALUE':"
        f" ok={b_out['ok']} fence={b_out.get('claimed_fence')}"
        f" interrupts={b_out.get('interrupts')}")
    show_ctl(rows_b, task, "    control rows at the tip after B:")
    check(ctl(rows_b, NULL_TASK_ID, -4) == "B-VALUE"
          and ctl(rows_b, task, -4) == ["B-VALUE"],
          "the survivor's human input is the durable control data at that checkpoint",
          f"NULL/__resume__={ctl(rows_b, NULL_TASK_ID, -4)!r}"
          f"  task/__resume__={ctl(rows_b, task, -4)!r}")

    # stage 3 -- the zombie thaws and issues the write it was holding.
    (tmp / "thaw").write_text("go")
    os.kill(pid, signal.SIGCONT)
    try:
        pa.wait(timeout=120)
    except subprocess.TimeoutExpired:
        pa.kill()
        pa.wait()
        FAILURES.append("worker A deadlocked after thaw")
    a_out = read_out(a_cfg, "A")
    rows_a = writes_at(conninfo, tid, cp)
    snap_a = chain(conninfo, tid)
    say(f"  stage 3  worker A SIGCONT'd and allowed to write: ok={a_out['ok']}"
        f" error={a_out.get('error')} sqlstate={a_out.get('sqlstate')}"
        f" interrupts={a_out.get('interrupts')}")
    echo_stderr(a_cfg, "A")
    show_ctl(rows_a, task, "    control rows at the tip after the zombie wrote:")

    if not fenced:
        check(ctl(rows_a, NULL_TASK_ID, -4) == "A-VALUE"
              and ctl(rows_a, task, -4) == ["A-VALUE"],
              "THE OVERWRITE: the stale worker's __resume__ REPLACED the survivor's."
              " Control data, not transcript",
              f"NULL/__resume__ B-VALUE -> {ctl(rows_a, NULL_TASK_ID, -4)!r};"
              f" task/__resume__ ['B-VALUE'] -> {ctl(rows_a, task, -4)!r}")
        check(len(rows_a) == len(rows_b),
              "it REPLACED rather than added: same conflict key, same row count",
              f"{len(rows_b)} rows before, {len(rows_a)} after")
        check(a_out["ok"] and b_out["ok"],
              "both workers believe they succeeded; neither raised, neither logged")
    else:
        stale = claim(conninfo, tid, 1, "C")
        check(b_out.get("claimed_fence") == 2 and stale is None,
              "worker B's claim is a compare-and-set: 1 -> 2 in one statement, and a"
              " later claim expecting 1 gets nothing",
              f"B claimed {b_out.get('claimed_fence')}; stale re-claim -> {stale}")
        check(not a_out["ok"], "worker A's run terminated instead of writing",
              str(a_out.get("error")))
        check(a_out.get("sqlstate") == "22012",
              "the refusal came from POSTGRES, not a Python raise -- LOAD-BEARING:"
              " a client-side raise inside a pipeline commits the queued write anyway",
              f"sqlstate={a_out.get('sqlstate')}")
        check(ctl(rows_a, NULL_TASK_ID, -4) == "B-VALUE"
              and ctl(rows_a, task, -4) == ["B-VALUE"],
              "the fence PREVENTS the overwrite: the survivor's control data is"
              " untouched, on the DO UPDATE statement as well as DO NOTHING",
              f"NULL/__resume__={ctl(rows_a, NULL_TASK_ID, -4)!r}"
              f"  task/__resume__={ctl(rows_a, task, -4)!r}")
        check(read_fence(conninfo, tid) == 2, "the fence row advanced exactly once")

    check(not forks(snap_a) and len(snap_a) == len(snap_b) == len(snap0),
          "no fork, no checkpoint added: this damage is NOT the dropped-write"
          " mechanism proof.py shows",
          f"{len(snap0)} checkpoints at stage 0, {len(snap_b)} after B,"
          f" {len(snap_a)} after A; forks={len(forks(snap_a))}")

    # stage 4 -- the consequence. Identical code in both halves; only A was fenced.
    say("  stage 4  a LATER worker resumes the second interrupt with 'SECOND'."
        "\n           Identical in both halves -- unfenced, no checkpoint_id pinned,"
        " writes nothing first.")
    try:
        with psycopg.connect(conninfo, autocommit=True) as conn:
            st = build(PostgresSaver(conn)).invoke(
                Command(resume="SECOND"), {"configurable": {"thread_id": tid}},
                durability="sync")
        log = st.get("log")
    except BaseException as e:  # noqa: BLE001
        log = f"<{type(e).__name__}: {e}>"
    say(f"           the run COMPLETED. final log = {log}")
    if not fenced:
        check(log == ["start", "A-VALUE|SECOND"],
              "the surviving run finished on a human input NEVER GIVEN TO IT: the"
              " frozen worker's 'A-VALUE', not the live worker's 'B-VALUE'",
              f"log={log}")
    else:
        check(log == ["start", "B-VALUE|SECOND"],
              "fenced, the run finishes on the input that was actually supplied to"
              " the live worker",
              f"log={log}")


def part2b(conninfo: str, trials: int = 5) -> None:
    """Answers the obvious objection to Part 2: that the ordering was engineered.

    Same thread, same two resume values, but no gate, no SIGSTOP, no fence --
    two processes started together and left to race. Measured, not asserted,
    except for the one thing that is not a matter of timing: that both
    processes address the SAME row.
    """
    say("\n" + "=" * 78)
    say("PART 2b -- REAL GRAPH RUN, NO INSTRUMENTATION: the collision without a gate")
    say("=" * 78)
    say("  Part 2's gate only decides WHICH worker writes last. Here: two processes")
    say("  spawned together (alternating which goes first), no gate, no SIGSTOP, no")
    say("  fence -- so the winner is decided by scheduling alone. The tally below")
    say("  varies from run to run and is therefore reported, not asserted; the one")
    say("  thing that is not a matter of timing IS asserted -- that the two")
    say("  processes contend for ONE row rather than storing one each.")
    tally: dict[str, int] = {}
    one_row = True
    for i in range(trials):
        tid = f"ow-race-{i}"
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="overwrite-race-"))
        with psycopg.connect(conninfo, autocommit=True) as conn:
            build(PostgresSaver(conn)).invoke({"log": ["start"]},
                                              {"configurable": {"thread_id": tid}},
                                              durability="sync")
        cp = chain(conninfo, tid)[-1]["id"]
        # alternate the spawn order: the point is that the winner is an accident
        # of scheduling, not a property of either worker.
        order = ("A", "B") if i % 2 == 0 else ("B", "A")
        cfgs = {r: {"conninfo": conninfo, "thread_id": tid, "role": r,
                    "resume": f"{r}-VALUE", "out_file": str(tmp / f"{r}.json"),
                    "fence": None} for r in ("A", "B")}
        procs = [spawn(cfgs[r]) for r in order]
        for p in procs:
            try:
                p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                p.kill()
        outs = [read_out(cfgs[r], r) for r in ("A", "B")]
        rows = writes_at(conninfo, tid, cp)
        task = next((r["task_id"] for r in rows if r["channel"] == "__interrupt__"), "?")
        n = len([r for r in rows if r["task_id"] == NULL_TASK_ID and r["idx"] == -4])
        one_row = one_row and n == 1
        try:
            with psycopg.connect(conninfo, autocommit=True) as conn:
                st = build(PostgresSaver(conn)).invoke(
                    Command(resume="SECOND"), {"configurable": {"thread_id": tid}},
                    durability="sync")
            after = st.get("log")
        except BaseException as e:  # noqa: BLE001
            after = f"<{type(e).__name__}: {str(e)[:60]}>"
        won = str(ctl(rows, task, -4))
        tally[won] = tally.get(won, 0) + 1
        t0 = min(o.get("t_start", 0) for o in outs)
        span = "  ".join(
            f"{o['role']}:{o.get('t_start', 0) - t0:+.3f}->"
            f"{o.get('t_end', 0) - t0:+.3f}s/log={o.get('log')}" for o in outs)
        say(f"    trial {i}  spawned {order[0]} then {order[1]}"
            f"  rows at (NULL_TASK_ID,-4)={n}  kept={won}\n"
            f"             {span}\n"
            f"             then a later worker resumed with 'SECOND' -> {after}")
    say(f"  which worker's value the thread kept, over {trials} trials: {tally}")
    check(one_row,
          "two unsynchronised processes always contend for ONE row at the shared"
          " conflict key -- they never store one each",
          "so the collision needs no instrumentation; the gate in Part 2 only makes"
          " the winner deterministic")


# ---------------------------------------------------------------- worker mode ---
def worker(cfg: dict) -> int:
    out: dict = {"role": cfg["role"], "pid": os.getpid(), "ok": False,
                 "t_start": round(time.time(), 4)}
    try:
        if cfg["role"] == "B" and cfg.get("fence") is not None:
            out["claimed_fence"] = claim(cfg["conninfo"], cfg["thread_id"],
                                         cfg["fence"] - 1, "B")
            cfg["fence"] = out["claimed_fence"]
        conn = psycopg.connect(cfg["conninfo"], autocommit=True)
        if cfg.get("fence") is None:
            saver = GatedSaver(conn)
        else:
            saver = GatedFencedSaver(conn, cfg["thread_id"], cfg["fence"])
        if cfg.get("gate_thaw"):
            saver.gate_ready = cfg["gate_ready"]
            saver.gate_thaw = cfg["gate_thaw"]
        # The resume path writes NOTHING to the thread first: no update_state,
        # no fork, no pinned checkpoint_id.
        st = build(saver).invoke(Command(resume=cfg["resume"]),
                                 {"configurable": {"thread_id": cfg["thread_id"]}},
                                 durability="sync")
        out["ok"] = True
        out["log"] = st.get("log")
        out["interrupts"] = [str(i.value) for i in (st.get("__interrupt__") or ())]
    except BaseException as e:  # noqa: BLE001 - report, never mask
        out["error"] = f"{type(e).__name__}: {e}".strip()[:300]
        out["sqlstate"] = getattr(e, "sqlstate", None)
    out["t_end"] = round(time.time(), 4)
    pathlib.Path(cfg["out_file"]).write_text(json.dumps(out))
    return 0 if out["ok"] else 1


def main() -> int:
    global LOG
    import importlib.metadata as md
    LOG = open(HERE / "overwrite.out", "w", buffering=1)
    say("THE OVERWRITE PROOF -- the other ON CONFLICT clause, end to end")
    say(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}   python"
        f" {sys.version.split()[0]}\npinned: " + "  ".join(
            f"{p}=={md.version(p)}" for p in (
                "langgraph", "langgraph-checkpoint", "langgraph-checkpoint-postgres",
                "langchain-core", "psycopg")))
    conninfo = pg_conninfo("durable_agents_overwrite")
    setup_database(conninfo)
    with psycopg.connect(conninfo, autocommit=True) as c:
        say("postgres: " + c.execute("show server_version").fetchone()[0]
            + f"   pipeline support: {psycopg.capabilities.has_pipeline()}")
    say("companion to proof.py: that one shows the DROP, this one the OVERWRITE")

    part1(conninfo)
    part1b(conninfo)
    scenario(conninfo, fenced=False)
    part2b(conninfo)
    scenario(conninfo, fenced=True)

    say("\n" + "=" * 78)
    if FAILURES:
        say(f"VERDICT: FAILED -- {len(FAILURES)} assertion(s) did not hold")
        for f in FAILURES:
            say(f"  - {f}")
    else:
        say("VERDICT: PROVEN")
        say("  SQL      a write whose channels are ALL in WRITES_IDX_MAP takes")
        say("           UPSERT .. ON CONFLICT DO UPDATE, so the SECOND writer at a")
        say("           conflict key wins; an ordinary or mixed write at the same key")
        say("           takes DO NOTHING and the second writer is dropped. Measured")
        say("           through put_writes() itself, not inferred from the source.")
        say("  OFF      two OS processes on one thread_id: the SIGSTOP'd worker's")
        say("           __resume__ write REPLACED the live worker's, at the same")
        say("           checkpoint, with no fork and no added checkpoint -- so this is")
        say("           a different mechanism from the dropped write in proof.py. A")
        say("           later worker then resumed the run and finished it on a human")
        say("           input that was never given to it. All three workers exit 0.")
        say("  ON       proof/fence.py already blocks it: the guard sits at")
        say("           _cursor(pipeline=True), upstream of the statement choice, so")
        say("           the zombie is refused by Postgres (SQLSTATE 22012) on the DO")
        say("           UPDATE path exactly as on DO NOTHING, and the later worker")
        say("           finishes on the input actually supplied.")
    say("=" * 78)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        sys.exit(worker(json.loads(sys.argv[2])))
    sys.exit(main())
