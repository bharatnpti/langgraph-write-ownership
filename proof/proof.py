"""The zombie proof. One command, two halves, one verdict.

    spike/.venv/bin/python proof/proof.py

Half A (fencing OFF) is the deliverable: two OS processes, one thread_id, a real
Deep Agents loop. Worker A is SIGSTOP'd mid-super-step, worker B takes the
thread over and finishes, worker A is SIGCONT'd and writes.

Two tiers of claim, kept apart on purpose. TIER 1 is LangGraph's: the chain
forks and one worker's committed work goes unreachable while it reports success.
TIER 2 is deepagents': because `messages` is a DeltaChannel, the surviving
list is also MALFORMED. A plain MessagesState graph forks the same way and stays
internally consistent -- see README.

Half B is the same scenario with a per-write SQL fence, twice: once on the
pipeline path and once with supports_pipeline forced False.
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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "spike"))

import psycopg  # noqa: E402
from _harness import ScriptedChatModel, pg_conninfo  # noqa: E402
from deepagents import create_deep_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.base import WRITES_IDX_MAP  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from langgraph.checkpoint.postgres.base import (  # noqa: E402
    INSERT_CHECKPOINT_WRITES_SQL, UPSERT_CHECKPOINT_WRITES_SQL)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402

from fence import claim, install_fence, orphans, read_fence, setup_database  # noqa: E402

PY = str(HERE.parents[0] / "spike" / ".venv" / "bin" / "python")
LOG = open(HERE / "proof.out", "w", buffering=1)
FAILURES: list[str] = []

# A real LLM gives the two workers different continuations; scripting that is
# the faithful stand-in. A's 2nd turn calls tc_A2, B's calls tc_B2.
SCRIPT_A = [{"content": "looking up k1", "tool": {"id": "tc_A1", "key": "k1"}},
            {"content": "worker A turn 2", "tool": {"id": "tc_A2", "key": "k2"}},
            {"content": "final answer from A", "tool": None}]
SCRIPT_B = [{"content": "worker B turn 2", "tool": {"id": "tc_B2", "key": "k2"}},
            {"content": "final answer from B", "tool": None}]


def say(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.write(msg + "\n")


def check(ok: bool, label: str, detail: str = "") -> bool:
    say(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)
    return ok


@tool
def lookup(key: str) -> str:
    """Look up a key in the knowledge base."""
    return f"value-of-{key}"


class ProviderModel(ScriptedChatModel):
    """Stands in for a real model API: rejects a malformed message list the way
    Anthropic/OpenAI do (every tool_result needs its tool_use and vice versa).
    We have no API key, so this is OUR check, not a provider's 400."""

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        problems = orphans(messages)
        if problems:
            raise RuntimeError("provider would reject this request: " + problems[0])
        return super()._generate(messages, stop, run_manager, **kw)


def agent_for(conninfo: str, model=None):
    conn = psycopg.connect(conninfo, autocommit=True)
    saver = PostgresSaver(conn)
    return create_deep_agent(
        model=model or ScriptedChatModel(responses=[]),
        tools=[lookup],
        checkpointer=saver,
    )


def chain(conninfo: str, thread_id: str) -> list[dict]:
    with psycopg.connect(conninfo, autocommit=True, row_factory=psycopg.rows.dict_row) as c:
        return c.execute(
            "select checkpoint_id id, parent_checkpoint_id parent,"
            " metadata->>'step' step from checkpoints"
            " where thread_id=%s and checkpoint_ns='' order by checkpoint_id",
            (thread_id,),
        ).fetchall()


def forks(rows: list[dict]) -> list[tuple[str, list[str]]]:
    kids: dict[str, list[str]] = {}
    for r in rows:
        if r["parent"]:
            kids.setdefault(r["parent"], []).append(r["id"])
    return [(p, v) for p, v in kids.items() if len(v) > 1]


def stored_msg_at(conninfo: str, thread_id: str, cid: str):
    """Deserialize the messages writes durably stored at one checkpoint. For a
    DeltaChannel (deepagents' `messages`) this IS the state: no full-list blob
    is written, so reconstruction replays these rows along the parent chain.
    Returns [(task_id, [messages])]."""
    with psycopg.connect(conninfo, autocommit=True) as c:
        rows = c.execute(
            "select task_id, type, blob from checkpoint_writes where thread_id=%s"
            " and checkpoint_ns='' and checkpoint_id=%s and channel='messages'",
            (thread_id, cid),
        ).fetchall()
    serde = JsonPlusSerializer()
    out = []
    for task_id, t, b in rows:
        v = serde.loads_typed((t, b))
        out.append((task_id, v if isinstance(v, list) else [v]))
    return out


def stored_tool_ids(conninfo: str, thread_id: str) -> tuple[set[str], set[str]]:
    """Every tool_use id offered and every tool_result id answered ANYWHERE in
    the thread's durable state -- both checkpoint_writes rows and
    checkpoint_blobs (a DeltaChannel writes no blobs; a plain add_messages
    channel writes only blobs). Returns (offered, answered)."""
    serde, offered, answered = JsonPlusSerializer(), set(), set()
    with psycopg.connect(conninfo, autocommit=True) as c:
        rows = c.execute(
            "select type, blob from checkpoint_writes where thread_id=%s"
            " and channel='messages'", (thread_id,)).fetchall()
        rows += c.execute(
            "select type, blob from checkpoint_blobs where thread_id=%s"
            " and channel='messages' and blob is not null", (thread_id,)).fetchall()
    for t, b in rows:
        try:
            v = serde.loads_typed((t, bytes(b)))
        except Exception:  # noqa: BLE001 - a blob we cannot read offers nothing
            continue
        for m in (v if isinstance(v, list) else [v]):
            offered |= {c["id"] for c in (getattr(m, "tool_calls", None) or [])}
            if getattr(m, "tool_call_id", None):
                answered.add(m.tool_call_id)
    return offered, answered


def spawn(cfg: dict) -> subprocess.Popen:
    err = open(cfg["out_file"] + ".err", "w")
    return subprocess.Popen([PY, str(HERE / "worker.py"), json.dumps(cfg)], stderr=err)


def echo_stderr(cfg: dict, who: str) -> None:
    txt = pathlib.Path(cfg["out_file"] + ".err").read_text().strip()
    for line in txt.splitlines():
        say(f"    [{who} stderr] {line}")


def read_out(cfg: dict, who: str) -> dict:
    """Never die on a missing result file: a hung or hard-crashed worker must
    still produce a legible FAIL and let the verdict print."""
    p = pathlib.Path(cfg["out_file"])
    if p.exists():
        return json.loads(p.read_text())
    FAILURES.append(f"worker {who} produced no result file (hang or hard crash)")
    say(f"  !! worker {who} wrote no result file ({p}) -- hang or hard crash;"
        " recording a failure and continuing so the verdict still prints")
    return {"ok": False, "error": f"worker {who} produced no result file",
            "messages": [["AIMessage", "<no result file>", [], None]]}


def run_scenario(conninfo: str, thread_id: str, fenced: bool, no_pipeline: bool):
    """The one scenario, shared by both halves. Returns (snapshots, A, B)."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="proof-"))
    common = {"conninfo": conninfo, "thread_id": thread_id,
              "force_no_pipeline": no_pipeline}
    a_cfg = dict(common, role="A", script=SCRIPT_A, block_at=1,
                 ready_file=str(tmp / "ready"), thaw_file=str(tmp / "thaw"),
                 out_file=str(tmp / "a.json"), fence=1 if fenced else None)
    b_cfg = dict(common, role="B", script=SCRIPT_B, block_at=-1,
                 out_file=str(tmp / "b.json"), fence=2 if fenced else None)

    pa = spawn(a_cfg)
    end = time.time() + 60
    while not (tmp / "ready").exists() and time.time() < end:
        time.sleep(0.02)
    if not (tmp / "ready").exists():
        pa.kill()
        raise SystemExit("worker A never reached its 2nd model call; aborting")
    frozen_pid = int((tmp / "ready").read_text())
    os.kill(frozen_pid, signal.SIGSTOP)
    state = subprocess.run(["ps", "-o", "state=", "-p", str(frozen_pid)],
                           capture_output=True, text=True).stdout.strip()
    say(f"  worker A (pid {frozen_pid}) SIGSTOP'd inside its 2nd model call --"
        f" frozen, not killed (ps state={state!r}; T = stopped)")
    snap_frozen = chain(conninfo, thread_id)
    say(f"  chain at freeze: {len(snap_frozen)} checkpoints, tip step"
        f" {snap_frozen[-1]['step']} id ...{snap_frozen[-1]['id'][-8:]}")

    pb = spawn(b_cfg)
    try:
        pb.wait(timeout=180)
    except subprocess.TimeoutExpired:
        say("  !! worker B did not exit within 180s -- killing")
        pb.kill()
        FAILURES.append("worker B hung during the takeover")
    b_out = read_out(b_cfg, "B")
    say(f"  worker B took the thread over with invoke(None, config): ok={b_out['ok']}"
        f" fence={b_out.get('claimed_fence')}")
    snap_after_b = chain(conninfo, thread_id)

    (tmp / "thaw").write_text("go")
    os.kill(frozen_pid, signal.SIGCONT)
    say("  worker A SIGCONT'd -- the zombie thaws and writes")
    try:
        pa.wait(timeout=120)
    except subprocess.TimeoutExpired:
        say("  !! worker A did not exit within 120s (deadlock?) -- killing")
        pa.kill()
        pa.wait()
        FAILURES.append("worker A deadlocked after thaw")
    a_out = read_out(a_cfg, "A")
    say(f"  worker A finished: ok={a_out['ok']} error={a_out.get('error')}")
    echo_stderr(b_cfg, "B")
    echo_stderr(a_cfg, "A")
    return (snap_frozen, snap_after_b, chain(conninfo, thread_id)), a_out, b_out


def dump_thread(conninfo: str, thread_id: str, header: str):
    """Read the thread back and print it. get_state is a pure read (Q5)."""
    st = agent_for(conninfo).get_state({"configurable": {"thread_id": thread_id}})
    msgs = st.values.get("messages", [])
    say(header)
    for m in msgs:
        ids = [c["id"] for c in (getattr(m, "tool_calls", None) or [])]
        say(f"    {type(m).__name__:<14} {str(m.content)[:26]:<28} tool_calls="
            f"{ids or '-'} tool_call_id={getattr(m, 'tool_call_id', None) or '-'}")
    return msgs, st.next


def probe_resume(conninfo: str, thread_id: str) -> dict[str, bool]:
    """Both probes use the provider-faithful model, so a failure means the
    message list was rejected and a success means it was accepted.
    NOTE: this MUTATES the thread -- the next-turn probe commits a HumanMessage,
    deepagents' repair middleware runs, and a rejection leaves an __error__
    write. It therefore runs after every check(). See README."""
    rc = {"configurable": {"thread_id": thread_id}}
    got = {}
    for label, inp in (("invoke(None, config)", None),
                       ("next user turn", {"messages": [HumanMessage("continue")]})):
        try:
            st = agent_for(conninfo, ProviderModel(responses=[])).invoke(
                inp, rc, durability="sync")
            got[label] = True
            say(f"    {label:<20} -> ACCEPTED, {len(st['messages'])} messages"
                + (" (no model call: nothing pending)" if inp is None else ""))
        except BaseException as e:  # noqa: BLE001
            got[label] = False
            say(f"    {label:<20} -> {type(e).__name__}: {str(e)[:160]}")
    return got


def conflict_clauses(conninfo: str) -> None:
    """Measured, not asserted: put_writes picks its ON CONFLICT clause from the
    channels it is handed, and one of the two clauses is last-writer-wins."""
    say("\n  and the conflict clause put_writes picks (measured, not asserted):")
    say(f"    WRITES_IDX_MAP = {sorted(WRITES_IDX_MAP)}")
    with psycopg.connect(conninfo, autocommit=True) as c:
        for name, sql in (
            ("INSERT .. DO NOTHING   <- any channel outside that map ('messages')",
             INSERT_CHECKPOINT_WRITES_SQL),
            ("UPSERT .. DO UPDATE    <- only if ALL channels are in that map",
             UPSERT_CHECKPOINT_WRITES_SQL),
        ):
            tid = "pf-" + ("nothing" if "DO NOTHING" in sql else "update")
            for who in (b"survivor", b"zombie"):
                c.execute(sql, (tid, "", "cp", "task", "", 0, "__interrupt__",
                                "text", who))
            got = c.execute("select blob from checkpoint_writes where thread_id=%s",
                            (tid,)).fetchall()
            c.execute("delete from checkpoint_writes where thread_id=%s", (tid,))
            say(f"    {name}\n      two writers, same conflict key -> survivor is"
                f" {bytes(got[0][0]).decode()!r}")
    say("    so a zombie's __interrupt__/__resume__/__error__ write OVERWRITES the")
    say("    survivor's instead of being dropped. Mechanism measured above; the")
    say("    end-to-end scenario is untested -- see README 'What this does not prove'.")


def preflight(conninfo: str) -> None:
    """Negative controls for the guard's SQL, run against the live server rather
    than argued from the manual. See README 'Why the guard looks like that'."""
    say("\npre-flight -- the guard's SQL, and two shapes that look right and are not")
    sub = "(select count(*)::int from proof_fence where thread_id='pf' and fence=%d)"
    probes = (
        ("CASE WHEN <sub>=1 THEN 1 ELSE 1/0 END, fence MATCHES",
         f"select case when {sub % 1}=1 then 1 else 1/0 end"),
        ("CASE WHEN true THEN 1 ELSE 1/0 END (constant condition)",
         "select case when true then 1 else 1/0 end"),
        ("1 / (select count(*)...), fence MATCHES", f"select 1/{sub % 1}"),
        ("1 / (select count(*)...), fence STALE", f"select 1/{sub % 99}"),
    )
    got = []
    with psycopg.connect(conninfo, autocommit=True) as c:
        c.execute("insert into proof_fence values ('pf','x',1)"
                  " on conflict (thread_id) do update set fence=1")
        for label, sql in probes:
            try:
                got.append(str(c.execute(sql).fetchone()[0]))
            except psycopg.Error as e:
                got.append(f"{type(e).__name__} sqlstate={e.sqlstate}")
            say(f"    {label:<57} -> {got[-1]}")
        c.execute("delete from proof_fence where thread_id='pf'")
    check(got[2] == "1" and "DivisionByZero" in got[3],
          "the guard SQL returns 1 when the fence matches and raises 22012 when stale")
    conflict_clauses(conninfo)


def half_a(conninfo: str) -> None:
    tid = "half-a"
    say("\n" + "=" * 78)
    say("HALF A -- NEGATIVE CONTROL, FENCING OFF (the deliverable)")
    say("=" * 78)
    (snap_frozen, snap_after_b, snap_final), a_out, b_out = run_scenario(
        conninfo, tid, fenced=False, no_pipeline=False)
    fork_parent = snap_frozen[-1]["id"]

    say("\n  Assertion 1 -- TIER 1, general to LangGraph + PostgresSaver and in no"
        " way\n  specific to deepagents: the checkpoint chain FORKS and one"
        " worker's work is lost")
    f = forks(snap_final)
    check(len(f) == 1 and f[0][0] == fork_parent,
          "two distinct checkpoints share one parent_checkpoint_id",
          f"parent ...{fork_parent[-8:]} -> children "
          + ", ".join("..." + c[-8:] for c in (f[0][1] if f else [])))
    before = {r["id"] for r in snap_after_b}
    kids = f[0][1] if f else []
    b_kid = [c for c in kids if c in before]
    a_kid = [c for c in kids if c not in before]
    check(len(b_kid) == 1 and len(a_kid) == 1,
          "one child written by worker B, one by the thawed zombie A",
          f"B ...{b_kid[0][-8:]} / A ...{a_kid[0][-8:]}" if b_kid and a_kid else "")
    check(a_out["ok"] and b_out["ok"],
          "BOTH workers believe they succeeded; neither saw an error")

    by_id, reach, cur = {r["id"]: r for r in snap_final}, set(), snap_final[-1]
    while cur:
        reach.add(cur["id"])
        cur = by_id.get(cur["parent"])
    lost = [r["id"] for r in snap_after_b if r["id"] not in reach]
    b_tip = snap_after_b[-1]["id"]
    check(bool(lost) and b_tip in lost,
          "worker B's committed work -- its terminal checkpoint included -- is no"
          " longer reachable from the tip",
          f"{len(lost)} of B's {len(snap_after_b)} checkpoints orphaned; B's terminal"
          f" checkpoint ...{b_tip[-8:]} is"
          f" {'one of them' if b_tip in lost else 'NOT among them'}")

    say("\n  Assertion 2 -- TIER 2, specific to deepagents' DeltaChannel transcript:")
    say("  the surviving branch's message list is MALFORMED, not merely superseded")
    stored = stored_msg_at(conninfo, tid, fork_parent)
    ids = sorted({c["id"] for _, ms in stored for m in ms
                  for c in (getattr(m, "tool_calls", None) or [])})
    offered, answered = stored_tool_ids(conninfo, tid)
    say(f"    both workers ran the SAME model task at the fork parent and both wrote"
        f" messages there;\n    rows actually stored: {len(stored)}"
        f"  task_ids={[t[:8] for t, _ in stored]}  tool_calls={ids}")
    say(f"    every tool_use id stored ANYWHERE in the thread (writes + blobs):"
        f" {sorted(offered)}\n    every tool_result id stored anywhere:"
        f"          {sorted(answered)}")
    check(len(stored) == 1 and ids == ["tc_B2"]
          and "tc_A2" not in offered and "tc_A2" in answered,
          "the zombie's AIMessage was not superseded, it was ANNIHILATED: nothing in"
          " the thread offers tc_A2, yet a tool_result answering it is durable",
          "one messages row survives at the fork parent (B's, offering tc_B2); the"
          " AIMessage offering tc_A2 is in no write row and no blob, while A's"
          " ToolMessage for tc_A2 sits on A's branch -- reachable only if A ran the"
          " same task at the same checkpoint and its put_writes was dropped")
    msgs, nxt = dump_thread(conninfo, tid, "    thread as read back from the tip:")
    problems = orphans(msgs)
    check(any("has no matching" in p for p in problems),
          "the list read back from the tip contains a tool_result whose tool_use is"
          " nowhere in it",
          next((p for p in problems if "has no matching" in p),
               "no orphaned tool_result found"))
    for p in problems:
        if "has no matching" not in p:
            say(f"         + also, but NOT what the assertion above tests: {p}")
    say(f"    worker B returned to its caller: {b_out['messages'][-1][1]!r}")
    say(f"    the thread's tip now says:       {str(msgs[-1].content)[:40]!r}")
    say("    A plain StateGraph on MessagesState forks the chain identically but")
    say("    leaves each branch internally consistent -- see README 'Two tiers'.")

    say("\n  Assertion 3 -- is it unresumable? (measured and reported, NOT asserted)")
    say(f"    .next at the tip = {nxt}  (LangGraph thinks the run finished)")
    say("    the probes below MUTATE the thread; they run after every check() above")
    probe_resume(conninfo, tid)
    say("    Plainly: NOT mechanically unresumable -- LangGraph reads, replays and")
    say("    resumes the forked thread without complaint. What is destroyed is the")
    say("    thread's USABILITY at the next model call, durably. Shown with our own")
    say("    validator, not a real provider's 400 (no API key here).")

    say("\n  Assertion 4 -- the independent witness: deepagents' OWN repair"
        " middleware (measured, NOT asserted)")
    msgs2, _ = dump_thread(
        conninfo, tid, "    the thread after that next-turn probe:")
    repaired = any("was cancelled" in str(m.content) for m in msgs2)
    still_bad = [p for p in orphans(msgs2) if "has no matching" in p]
    say(f"    PatchToolCallsMiddleware.before_agent fired, rewrote the list with"
        f" RemoveMessage(REMOVE_ALL_MESSAGES)\n    and injected a synthetic"
        f" ToolMessage for the dangling tc_B2: {repaired}")
    say(f"    that repair is committed to the thread, and the orphaned tool_result"
        f" SURVIVED it: {bool(still_bad)}")
    say("    deepagents itself calls half of this damage a defect, and cannot fix"
        " the other half.")


def half_b(conninfo: str, no_pipeline: bool) -> None:
    tid = "half-b-nopipe" if no_pipeline else "half-b"
    say("\n" + "=" * 78)
    say(f"HALF B -- FENCING ON  ({'supports_pipeline forced False' if no_pipeline else 'pipeline path'})")
    say("=" * 78)
    install_fence(conninfo, tid, "A")
    (snap_frozen, snap_after_b, snap_final), a_out, b_out = run_scenario(
        conninfo, tid, fenced=True, no_pipeline=no_pipeline)

    stale = claim(conninfo, tid, 1, "C")  # a late claimant with a stale expectation
    check(b_out.get("claimed_fence") == 2 and stale is None,
          "worker B's claim is a compare-and-set: it advanced the fence 1 -> 2, and a"
          " later claim expecting 1 gets nothing",
          f"B claimed {b_out.get('claimed_fence')}; the stale re-claim returned {stale}")
    check(not forks(snap_final), "the chain did NOT fork",
          f"{len(forks(snap_final))} forked parents")
    added = [r["id"] for r in snap_final if r["id"] not in {x["id"] for x in snap_after_b}]
    check(not added, "the thawed zombie's writes were refused by the DATABASE",
          f"{len(added)} checkpoint rows added after the takeover")
    check(not a_out["ok"], "worker A's run terminated instead of committing",
          f"{a_out.get('error')}")
    check(a_out.get("sqlstate") == "22012",
          "the refusal came from POSTGRES, not from a Python raise -- LOAD-BEARING,"
          " see README: the only assertion here a client-side fence fails",
          f"sqlstate={a_out.get('sqlstate')} error={a_out.get('error')}")
    check(read_fence(conninfo, tid) == 2, "the fence row advanced exactly once")
    msgs, _ = dump_thread(conninfo, tid, "  worker B's thread, read back:")
    shape = [type(m).__name__ for m in msgs]
    check(shape == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
                    "ToolMessage", "AIMessage"] and not orphans(msgs),
          "worker B's result is intact and well-formed",
          f"{len(msgs)} messages, shape {'as expected' if len(msgs) == 6 else shape},"
          f" orphans: {'; '.join(orphans(msgs)) or 'none'}")
    check(str(msgs[-1].content).endswith("from B"),
          "the tip is worker B's answer", repr(str(msgs[-1].content)))
    say("  still resumable (these probes MUTATE the thread; they run last):")
    got = probe_resume(conninfo, tid)
    check(got.get("next user turn") is True,
          "the thread accepts a next turn: a fresh model call on the surviving list"
          " is not rejected",
          f"invoke(None)={got.get('invoke(None, config)')}"
          f" next-turn={got.get('next user turn')}")


def main() -> int:
    import importlib.metadata as md
    say("THE ZOMBIE PROOF -- durable-agents Phase 1")
    say(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}   python {sys.version.split()[0]}"
        "\npinned: " + "  ".join(f"{p}=={md.version(p)}" for p in (
            "langgraph", "langgraph-checkpoint", "langgraph-checkpoint-postgres",
            "deepagents", "langchain-core", "psycopg")))
    conninfo = pg_conninfo("durable_agents_proof")
    setup_database(conninfo)
    with psycopg.connect(conninfo, autocommit=True) as c:
        say("postgres: " + c.execute("show server_version").fetchone()[0]
            + f"   pipeline support: {psycopg.capabilities.has_pipeline()}")
    say("scenario: 1 thread, 2 OS processes, deepagents react loop, durability='sync'")

    preflight(conninfo)
    half_a(conninfo)
    half_b(conninfo, no_pipeline=False)
    half_b(conninfo, no_pipeline=True)

    say("\n" + "=" * 78)
    if FAILURES:
        say(f"VERDICT: FAILED -- {len(FAILURES)} assertion(s) did not hold")
        for f in FAILURES:
            say(f"  - {f}")
    else:
        say("VERDICT: PROVEN, in two tiers")
        say("  OFF, TIER 1 -- general to LangGraph + PostgresSaver, nothing to do with")
        say("       deepagents: two workers on one thread_id fork the checkpoint chain;")
        say("       the loser's put_writes is silently discarded while it reports")
        say("       success; its committed work, terminal checkpoint included, is")
        say("       unreachable from the tip; the answer it returned to its caller")
        say("       contradicts the thread. Both workers exit 0. Nothing warns.")
        say("  OFF, TIER 2 -- specific to deepagents' DeltaChannel transcript: the")
        say("       surviving branch's message list is MALFORMED -- a tool_result whose")
        say("       tool_use was silently discarded and is stored NOWHERE, alongside a")
        say("       tool_use whose result is on the abandoned branch. A plain")
        say("       MessagesState graph forks identically but leaves each branch")
        say("       internally consistent, because its full list lives in its own blob.")
        say("  ON:  the same zombie's writes were refused by Postgres, the chain did not")
        say("       fork, and the survivor's thread is intact and usable -- on both the")
        say("       pipeline path and the supports_pipeline=False fallback.")
    say("=" * 78)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
