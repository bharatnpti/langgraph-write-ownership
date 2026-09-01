"""
Q3 ADVERSARIAL VERIFY 1 -- lens: SWALLOWING AND RETRIES.

Claim under attack: "an exception raised from a saver write RELIABLY REACHES the
caller, so fencing-by-exception is a safe mechanism."

This script hunts every path that retries, wraps, backgrounds, defers or
DISCARDS a saver-write exception, and proves each one with a test.

Everything printed with OBSERVED: was executed in this process.
Everything printed with SOURCE: was read in the installed langgraph source.

Run:
  spike/.venv/bin/python \
    spike/q3_verify1_swallow_paths.py
"""

from __future__ import annotations
import pathlib

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "spike"))

import asyncio
import concurrent.futures
import gc
import logging
import operator
import signal
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from typing import Annotated, Any, Sequence, TypedDict

import langgraph
import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime, RunControl
from langgraph.types import interrupt

from _harness import pg_conninfo

CONN = pg_conninfo("q3v1_swallow")
RUN = uuid.uuid4().hex[:8]

OUT = open(
    str(pathlib.Path(__file__).resolve().parents[1] / "spike") + "/"
    "q3_verify1_swallow_paths.out",
    "w",
    buffering=1,
)


def p(*a: Any) -> None:
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    OUT.write(msg + "\n")
    OUT.flush()


def hdr(title: str) -> None:
    p("")
    p("=" * 78)
    p(title)
    p("=" * 78)


# --------------------------------------------------------------- exceptions --
class StaleOwner(Exception):
    """our fence: 'you lost the lease; this write is refused'"""


class SiblingBoom(Exception):
    """a different background failure, used to test sibling replacement"""


class NodeBoom(Exception):
    """a plain node failure"""


# ---------------------------------------------------------------- fixtures --
SIDE: list[str] = []


class S(TypedDict):
    log: Annotated[list[str], operator.add]


def _n(name: str, sleep: float = 0.0, raises: BaseException | None = None):
    def fn(state: S) -> S:
        if sleep:
            time.sleep(sleep)
        SIDE.append(name)
        if raises is not None:
            raise raises
        return {"log": [name]}

    return fn


def _an(name: str, sleep: float = 0.0, raises: BaseException | None = None):
    async def fn(state: S) -> S:
        if sleep:
            await asyncio.sleep(sleep)
        SIDE.append(name)
        if raises is not None:
            raise raises
        return {"log": [name]}

    return fn


def linear(async_=False):
    g = StateGraph(S)
    mk = _an if async_ else _n
    for n in ("n1", "n2", "n3"):
        g.add_node(n, mk(n))
    g.add_edge(START, "n1")
    g.add_edge("n1", "n2")
    g.add_edge("n2", "n3")
    g.add_edge("n3", END)
    return g.compile


def race_graph(async_=False, bad_exc: BaseException | None = None, bad_sleep=0.25):
    """START -> n1 -> {good, bad} (same superstep) -> join -> END

    `good` finishes instantly (its put_writes can be rejected while `bad` is
    still running); `bad` then raises. This makes the rejection PENDING in the
    background executor at the exact moment a real node error starts
    propagating -- the race BackgroundExecutor.__exit__ guards with
    `if exc_type is None`.
    """
    mk = _an if async_ else _n
    g = StateGraph(S)
    g.add_node("n1", mk("n1"))
    g.add_node("good", mk("good"))
    g.add_node("bad", mk("bad", sleep=bad_sleep, raises=bad_exc or NodeBoom("node blew up")))
    g.add_node("join", mk("join"))
    g.add_edge(START, "n1")
    g.add_edge("n1", "good")
    g.add_edge("n1", "bad")
    g.add_edge("good", "join")
    g.add_edge("bad", "join")
    g.add_edge("join", END)
    return g.compile


# ------------------------------------------------------------- trap savers --
class TrapMixin:
    def arm(
        self,
        *,
        trap: str,
        predicate,
        once: bool = False,
        exc_factory=None,
        delay: float = 0.0,
    ) -> None:
        self.trap = trap  # "put" | "put_writes" | "all"
        self.predicate = predicate
        self.once = once
        self.delay = delay
        self.exc_factory = exc_factory or (lambda m, d, n: StaleOwner(f"fence lost: {m} {d}"))
        self.calls: list[str] = []
        self.rejected: list[str] = []
        self.by_key: dict[str, int] = {}
        self.nth = 0

    def _hit(self, method: str, desc: str, key: str, **kw) -> None:
        self.calls.append(f"{method}({desc})")
        self.by_key[key] = self.by_key.get(key, 0) + 1
        if self.trap not in (method, "all"):
            return
        if self.once and self.rejected:
            return
        if self.predicate(**kw):
            self.nth += 1
            if self.delay:
                time.sleep(self.delay)
            self.rejected.append(f"{method}({desc})")
            self.calls[-1] += "  <-- REJECTED"
            raise self.exc_factory(method, desc, self.nth)


def _pd(metadata) -> str:
    return f"step={metadata.get('step')},src={metadata.get('source')}"


def _wd(writes, task_id) -> str:
    return f"task={task_id[:8]},chans={[c for c, _ in writes]}"


class TrapSaver(TrapMixin, PostgresSaver):
    def put(self, config, checkpoint, metadata, new_versions):
        self._hit(
            "put",
            _pd(metadata),
            f"put:{checkpoint['id']}",
            metadata=metadata,
            config=config,
            checkpoint=checkpoint,
        )
        return super().put(config, checkpoint, metadata, new_versions)

    def put_writes(self, config, writes, task_id, task_path=""):
        self._hit(
            "put_writes",
            _wd(writes, task_id),
            f"pw:{task_id}",
            writes=writes,
            task_id=task_id,
            config=config,
        )
        return super().put_writes(config, writes, task_id, task_path)


class AsyncTrapSaver(TrapMixin, AsyncPostgresSaver):
    async def aput(self, config, checkpoint, metadata, new_versions):
        self._hit(
            "put",
            _pd(metadata),
            f"put:{checkpoint['id']}",
            metadata=metadata,
            config=config,
            checkpoint=checkpoint,
        )
        return await super().aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        self._hit(
            "put_writes",
            _wd(writes, task_id),
            f"pw:{task_id}",
            writes=writes,
            task_id=task_id,
            config=config,
        )
        return await super().aput_writes(config, writes, task_id, task_path)


# predicates
def always(**kw) -> bool:
    return True


def writes_of(node: str):
    """reject the put_writes produced BY a given node (matched on its value)"""

    def f(writes=None, **kw):
        return writes is not None and any(node in repr(v) for _, v in writes)

    return f


def ns_nonempty(**kw) -> bool:
    """reject only writes inside a subgraph namespace"""
    cfg = kw.get("config") or {}
    ns = (cfg.get("configurable") or {}).get("checkpoint_ns", "")
    return bool(ns)


# ------------------------------------------------------------------- utils --
@contextmanager
def alarm(seconds: int, label: str):
    def boom(signum, frame):
        raise TimeoutError(f"HARD TIMEOUT {seconds}s in {label}")

    old = signal.signal(signal.SIGALRM, boom)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def chain(e: BaseException) -> list[str]:
    out, seen, stack = [], set(), [e]
    while stack:
        cur = stack.pop(0)
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        out.append(f"{type(cur).__module__}.{type(cur).__name__}")
        for nxt in (cur.__cause__, cur.__context__):
            if nxt is not None:
                stack.append(nxt)
        if isinstance(cur, BaseExceptionGroup):
            stack.extend(cur.exceptions)
    return out


def has_stale(e: BaseException | None) -> bool:
    return e is not None and any(n.endswith(".StaleOwner") for n in chain(e))


# ------------------------------------------------- global evidence channels --
CAPTURED_LOGS: list[str] = []
CAPTURED_ASYNCIO: list[str] = []
CAPTURED_THREAD: list[str] = []
CAPTURED_UNRAISABLE: list[str] = []


class _Cap(logging.Handler):
    def emit(self, record):
        try:
            CAPTURED_LOGS.append(f"{record.name}:{record.levelname}:{record.getMessage()}")
            if record.exc_info and record.exc_info[1] is not None:
                CAPTURED_LOGS[-1] += f" exc={type(record.exc_info[1]).__name__}"
        except Exception:
            pass


logging.getLogger().addHandler(_Cap())
logging.getLogger().setLevel(logging.DEBUG)


def _thook(args):
    CAPTURED_THREAD.append(f"{type(args.exc_value).__name__}: {args.exc_value}")


threading.excepthook = _thook


def _uhook(args):
    CAPTURED_UNRAISABLE.append(f"{type(args.exc_value).__name__}: {args.exc_value}")


sys.unraisablehook = _uhook


def _aio_handler(loop, context):
    exc = context.get("exception")
    CAPTURED_ASYNCIO.append(
        f"{context.get('message')} exc={type(exc).__name__ if exc else None}: {exc}"
    )


def clear_evidence() -> None:
    CAPTURED_LOGS.clear()
    CAPTURED_ASYNCIO.clear()
    CAPTURED_THREAD.clear()
    CAPTURED_UNRAISABLE.clear()


def stale_evidence() -> list[str]:
    """any out-of-band record of a StaleOwner anywhere the wrapper could see"""
    out = []
    for tag, lst in (
        ("logging", CAPTURED_LOGS),
        ("asyncio-handler", CAPTURED_ASYNCIO),
        ("threading.excepthook", CAPTURED_THREAD),
        ("sys.unraisablehook", CAPTURED_UNRAISABLE),
    ):
        for m in lst:
            if "StaleOwner" in m or "fence lost" in m:
                out.append(f"{tag}: {m[:160]}")
    return out


# ------------------------------------------------------------------ results --
RESULTS: list[tuple[str, str, int, str, bool, str]] = []


def record(case: str, dur: str, rejections: int, caller: str, surfaced: bool, verdict: str):
    RESULTS.append((case, dur, rejections, caller, surfaced, verdict))


def classify(case: str, dur: str, saver, exc: BaseException | None, note: str = "") -> str:
    ev = stale_evidence()
    surfaced = has_stale(exc)
    n = len(saver.rejected)
    if n == 0:
        verdict = "FENCE-NEVER-CONSULTED"
    elif surfaced:
        verdict = "SURFACED"
    elif ev:
        verdict = "DROPPED-but-LOGGED"
    else:
        verdict = "DROPPED-SILENT"
    p(f"  OBSERVED: rejections={n} rejected={saver.rejected}")
    p(
        f"  OBSERVED: caller exception = "
        f"{type(exc).__name__ if exc else None}: {str(exc)[:90]!r}"
    )
    p(f"  OBSERVED: exception chain = {chain(exc) if exc else []}")
    p(f"  OBSERVED: StaleOwner recoverable from caller's exception = {surfaced}")
    p(f"  OBSERVED: out-of-band records mentioning the rejection = {ev if ev else 'NONE'}")
    p(f"  OBSERVED: nodes that ran = {SIDE}")
    p(f"  VERDICT[{case}] = {verdict} {note}")
    record(case, dur, n, type(exc).__name__ if exc else "None", surfaced, verdict)
    return verdict


# -------------------------------------------------------------- sync runner --
def run_sync(
    *,
    case: str,
    durability: str,
    trap: str,
    predicate=always,
    graph_factory=None,
    mode: str = "invoke",
    once: bool = False,
    exc_factory=None,
    abandon_after: int | None = None,
    input_=None,
):
    """returns (saver, exception_seen_by_caller)"""
    SIDE.clear()
    clear_evidence()
    thread = f"{RUN}-{case}"
    gf = graph_factory or linear()
    p("")
    p(f"--- CASE {case}  (durability={durability}, trap={trap}, mode={mode})")
    exc: BaseException | None = None
    with TrapSaver.from_conn_string(CONN) as saver:
        saver.arm(trap=trap, predicate=predicate, once=once, exc_factory=exc_factory)
        graph = gf(checkpointer=saver)
        cfg = {"configurable": {"thread_id": thread}}
        try:
            with alarm(60, case):
                if mode == "invoke":
                    graph.invoke(input_ or {"log": ["in"]}, cfg, durability=durability)
                elif mode == "stream":
                    for _ in graph.stream(input_ or {"log": ["in"]}, cfg, durability=durability):
                        pass
                elif mode == "abandon":
                    it = graph.stream(input_ or {"log": ["in"]}, cfg, durability=durability)
                    seen = 0
                    for _ in it:
                        seen += 1
                        if seen >= (abandon_after or 1):
                            break
                    p(f"  OBSERVED: abandoned the generator after {seen} chunk(s)")
                    del it
                    gc.collect()
                    time.sleep(0.4)
                    gc.collect()
        except BaseException as e:  # noqa: BLE001
            exc = e
        for line in saver.calls:
            p(f"    {line}")
    return saver, exc


# ------------------------------------------------------------- async runner --
async def _arun(
    *,
    case: str,
    durability: str,
    trap: str,
    predicate=always,
    graph_factory=None,
    mode: str = "ainvoke",
    once: bool = False,
    exc_factory=None,
    cancel_after: float | None = None,
    abandon_after: int | None = None,
):
    SIDE.clear()
    clear_evidence()
    asyncio.get_running_loop().set_exception_handler(_aio_handler)
    thread = f"{RUN}-{case}"
    gf = graph_factory or linear(async_=True)
    p("")
    p(f"--- CASE {case}  (durability={durability}, trap={trap}, mode={mode})")
    exc: BaseException | None = None
    async with AsyncTrapSaver.from_conn_string(CONN) as saver:
        saver.arm(trap=trap, predicate=predicate, once=once, exc_factory=exc_factory)
        graph = gf(checkpointer=saver)
        cfg = {"configurable": {"thread_id": thread}}
        try:
            if mode == "ainvoke":
                await asyncio.wait_for(
                    graph.ainvoke({"log": ["in"]}, cfg, durability=durability), 60
                )
            elif mode == "astream":
                async for _ in graph.astream({"log": ["in"]}, cfg, durability=durability):
                    pass
            elif mode == "cancel":
                t = asyncio.ensure_future(
                    graph.ainvoke({"log": ["in"]}, cfg, durability=durability)
                )
                await asyncio.sleep(cancel_after or 0.1)
                t.cancel()
                await t
            elif mode == "abandon":
                agen = graph.astream({"log": ["in"]}, cfg, durability=durability)
                seen = 0
                async for _ in agen:
                    seen += 1
                    if seen >= (abandon_after or 1):
                        break
                p(f"  OBSERVED: abandoned the async generator after {seen} chunk(s)")
                await agen.aclose()
                await asyncio.sleep(0.4)
                gc.collect()
                await asyncio.sleep(0.1)
        except BaseException as e:  # noqa: BLE001
            exc = e
        for line in saver.calls:
            p(f"    {line}")
    # let asyncio GC any orphaned task so its exception handler fires
    await asyncio.sleep(0.2)
    gc.collect()
    await asyncio.sleep(0.1)
    return saver, exc


def run_async(**kw):
    return asyncio.run(_arun(**kw))


# ============================================================================
def main() -> None:
    p("Q3 ADVERSARIAL VERIFY 1 -- lens: SWALLOWING AND RETRIES")
    p(f"langgraph = {langgraph.__version__ if hasattr(langgraph, '__version__') else 'n/a'}")
    p(f"db = q3v1_swallow  run tag = {RUN}")
    with PostgresSaver.from_conn_string(CONN) as s:
        s.setup()

    # ---------------------------------------------------------------- S0 ----
    hdr("S0. SOURCE ENUMERATION -- every site that can drop a saver-write exc")
    for line in [
        "SOURCE: pregel/_loop.py:1697  self.submit = stack.enter_context(BackgroundExecutor(config));",
        "        then stack.push(self._suppress_interrupt).  ExitStack unwinds LIFO, so",
        "        _suppress_interrupt runs FIRST and BackgroundExecutor.__exit__ LAST.",
        "SOURCE: pregel/_executor.py:105-118  BackgroundExecutor.__exit__:",
        "          concurrent.futures.wait(pending); stack.__exit__(...);",
        "          `if exc_type is None:` -> only then loop over tasks and task.result()",
        "        => DROP SITE #1 (sync): any exception raised by a background saver write",
        "           is discarded whenever ANY exception is already propagating.",
        "SOURCE: pregel/_executor.py:190-205  AsyncBackgroundExecutor.__aexit__: same",
        "        `if exc_type is None:` guard, after `await asyncio.wait(tasks)`.",
        "        => DROP SITE #2 (async), and it re-raises only the FIRST failed task",
        "           in dict (submission) order -- no ExceptionGroup, siblings dropped.",
        "SOURCE: pregel/_executor.py:77-86  BackgroundExecutor.done(): calls task.result()",
        "        inside `except BaseException: pass` -- so a failed write task is silenced",
        "        at completion time and NOT logged; it is only retained in self.tasks.",
        "        => DROP SITE #3: nothing ever logs the sync-side rejection.",
        "SOURCE: pregel/_executor.py:57-59 / 143-146  submit(__reraise_on_exit__): saver",
        "        writes are submitted with the default True; only NODE futures use False",
        "        (_runner.py:774, :926). So reraise=False is NOT a saver drop site.",
        "SOURCE: pregel/_loop.py:466  `if self.durability != 'exit' and put_writes is not None:`",
        "        => DROP SITE #4: durability='exit' never calls put_writes at all,",
        "           so a fence implemented only in put_writes is never consulted.",
        "SOURCE: pregel/_loop.py:1324-1334  _suppress_interrupt: under durability='exit'",
        "        it calls _put_exit_delta_writes/_put_checkpoint/_put_pending_writes",
        "        EVEN WHEN exc_value is not None -- and those submits land in the same",
        "        BackgroundExecutor that is about to be exited with exc_type != None.",
        "        => DROP SITE #5: durability='exit' + any error == silent rejection.",
        "SOURCE: pregel/_loop.py:1336-1373  _suppress_interrupt returns True for a",
        "        top-level GraphInterrupt. ExitStack then passes (None,None,None) to the",
        "        outer BackgroundExecutor.__exit__, so a pending rejection DOES surface",
        "        on the interrupt path. (anti-drop site)",
        "SOURCE: pregel/_loop.py:1529-1546 / 1783-1801  _checkpointer_put_after_previous:",
        "        `try: prev.result() finally: put(...)` -- the finally means a rejected",
        "        put does NOT stop the next put from being attempted, and the prev",
        "        exception can be REPLACED by the later put's exception.",
        "SOURCE: pregel/_loop.py:1571-1573 / 1830-1836  schedule_error_handler waits on",
        "        _error_handler_write_futs with concurrent.futures.wait / asyncio.gather.",
        "        concurrent.futures.wait NEVER raises => sync error-handler path defers",
        "        the rejection to loop exit; asyncio.gather DOES raise.",
        "SOURCE: pregel/main.py:2985-2986 / 3461-3462  `if durability_=='sync':",
        "        loop._put_checkpoint_fut.result()` -- the ONLY place a put future is",
        "        actively awaited mid-run. put_writes futures are never awaited.",
        "SOURCE: pregel/main.py:3013-3015 / 3492-3494 GraphRecursionError / GraphDrained",
        "        are raised AFTER the `with loop` block, i.e. after the executor exit,",
        "        so a pending rejection wins over them (anti-drop site).",
        "SOURCE: checkpoint/postgres/__init__.py  _cursor(): no retry, no reconnect loop,",
        "        no resend. Nothing in langgraph/pregel retries a saver write.",
    ]:
        p(line)

    # ---------------------------------------------------------------- S1 ----
    hdr("S1. DROP SITE #1/#2: is the `if exc_type is None` race STRUCTURAL?")
    p("Same race in every durability x trap x sync/async combination.")
    p("Graph: n1 -> {good, bad} in ONE superstep; good's write is rejected while")
    p("bad is still running; bad then raises NodeBoom.")
    for dur in ("sync", "async", "exit"):
        for trap in ("put_writes", "put"):
            saver, exc = run_sync(
                case=f"s1-sync-{dur}-{trap}",
                durability=dur,
                trap=trap,
                predicate=writes_of("good") if trap == "put_writes" else always,
                graph_factory=race_graph(),
            )
            classify(f"s1-sync-{dur}-{trap}", dur, saver, exc)
    for dur in ("sync", "async", "exit"):
        for trap in ("put_writes", "put"):
            saver, exc = run_async(
                case=f"s1-async-{dur}-{trap}",
                durability=dur,
                trap=trap,
                predicate=writes_of("good") if trap == "put_writes" else always,
                graph_factory=race_graph(async_=True),
            )
            classify(f"s1-async-{dur}-{trap}", dur, saver, exc)

    # ---------------------------------------------------------------- S1b ---
    hdr("S1b. control: the SAME race but with interrupt() instead of a node error")
    p("_suppress_interrupt returns True for a top-level GraphInterrupt, which makes")
    p("ExitStack hand (None,None,None) to BackgroundExecutor.__exit__.")

    def interrupt_graph(async_=False):
        mk = _an if async_ else _n
        g = StateGraph(S)
        g.add_node("n1", mk("n1"))
        g.add_node("good", mk("good"))

        def bad(state: S):
            time.sleep(0.25)
            SIDE.append("bad")
            interrupt("stop here")

        g.add_node("bad", bad)
        g.add_node("join", mk("join"))
        g.add_edge(START, "n1")
        g.add_edge("n1", "good")
        g.add_edge("n1", "bad")
        g.add_edge("good", "join")
        g.add_edge("bad", "join")
        g.add_edge("join", END)
        return g.compile

    for dur in ("sync", "async"):
        saver, exc = run_sync(
            case=f"s1b-interrupt-{dur}",
            durability=dur,
            trap="put_writes",
            predicate=writes_of("good"),
            graph_factory=interrupt_graph(),
        )
        classify(f"s1b-interrupt-{dur}", dur, saver, exc, note="(interrupt, not error)")

    # ---------------------------------------------------------------- S2 ----
    hdr("S2. DROP SITE #5: durability='exit' + node error -> the exit put is dropped")
    p("Under durability='exit' the ONLY saver write happens inside")
    p("_suppress_interrupt, which runs while the node error is still propagating.")
    saver, exc = run_sync(
        case="s2-exit-put-nodeerror",
        durability="exit",
        trap="put",
        predicate=always,
        graph_factory=race_graph(bad_sleep=0.0),
    )
    classify("s2-exit-put-nodeerror", "exit", saver, exc)

    # ---------------------------------------------------------------- S3 ----
    hdr("S3. SIBLING REPLACEMENT / ExceptionGroup on the async + sync paths")
    p("Two background writes fail in the SAME superstep with DIFFERENT exceptions.")
    p("Which one reaches the caller? Is it an ExceptionGroup?")

    def stale_then_sibling(method, desc, n):
        """1st rejected write -> StaleOwner, 2nd -> an unrelated failure"""
        return (
            StaleOwner(f"fence lost: {method} {desc}")
            if n == 1
            else SiblingBoom(f"unrelated background failure: {method} {desc}")
        )

    def sibling_then_stale(method, desc, n):
        """1st rejected write -> an unrelated failure, 2nd -> StaleOwner"""
        return (
            SiblingBoom(f"unrelated background failure: {method} {desc}")
            if n == 1
            else StaleOwner(f"fence lost: {method} {desc}")
        )

    def fanout3(async_=False):
        mk = _an if async_ else _n
        g = StateGraph(S)
        g.add_node("n1", mk("n1"))
        for n in ("good", "other"):
            g.add_node(n, mk(n))
        g.add_node("join", mk("join"))
        g.add_edge(START, "n1")
        for n in ("good", "other"):
            g.add_edge("n1", n)
            g.add_edge(n, "join")
        g.add_edge("join", END)
        return g.compile

    def both(writes=None, **kw):
        return writes is not None and any(
            ("good" in repr(v) or "other" in repr(v)) for _, v in writes
        )

    for order, order_fn in (("stale1st", stale_then_sibling), ("stale2nd", sibling_then_stale)):
      for tag, runner, gf in (
        ("sync", run_sync, fanout3()),
        ("async", run_async, fanout3(async_=True)),
      ):
        for dur in ("sync", "async"):
            saver, exc = runner(
                case=f"s3-{order}-{tag}-{dur}",
                durability=dur,
                trap="put_writes",
                predicate=both,
                graph_factory=gf,
                exc_factory=order_fn,
            )
            p(f"  OBSERVED: is ExceptionGroup = {isinstance(exc, BaseExceptionGroup)}")
            p(f"  OBSERVED: SiblingBoom in chain = "
              f"{any(n.endswith('.SiblingBoom') for n in chain(exc)) if exc else False}")
            classify(f"s3-{order}-{tag}-{dur}", dur, saver, exc, note=f"rejection order: {order}")

    # ---------------------------------------------------------------- S4 ----
    hdr("S4. RETRIES / RESENDS: does anything call the same saver write twice?")
    p("once=True: the trap rejects the FIRST matching write and allows everything")
    p("afterwards. If any layer retried, we would see a 2nd call for the same key.")
    for dur in ("sync", "async"):
        saver, exc = run_sync(
            case=f"s4-once-pw-{dur}",
            durability=dur,
            trap="put_writes",
            predicate=writes_of("n1"),
            once=True,
        )
        dupes = {k: v for k, v in saver.by_key.items() if v > 1}
        p(f"  OBSERVED: per-(method,key) call counts = {saver.by_key}")
        p(f"  OBSERVED: keys called more than once = {dupes if dupes else 'NONE'}")
        classify(f"s4-once-pw-{dur}", dur, saver, exc, note=f"dupes={dupes}")
    # psycopg-shaped failure: does a connection error trigger reconnect/resend?
    saver, exc = run_sync(
        case="s4-operationalerror",
        durability="sync",
        trap="put",
        predicate=always,
        once=True,
        exc_factory=lambda m, d, n: psycopg.OperationalError(
            f"connection lost during {m} {d}"
        ),
    )
    p(f"  OBSERVED: per-(method,key) call counts = {saver.by_key}")
    p(f"  OBSERVED: caller got {type(exc).__name__}: {str(exc)[:80]!r}")
    p(f"  OBSERVED: any key called twice = "
      f"{ {k: v for k, v in saver.by_key.items() if v > 1} }")
    record(
        "s4-operationalerror",
        "sync",
        len(saver.rejected),
        type(exc).__name__ if exc else "None",
        exc is not None,
        "NO-RETRY" if not any(v > 1 for v in saver.by_key.values()) else "RETRIED",
    )

    # ---------------------------------------------------------------- S5 ----
    hdr("S5. CANCELLATION with a rejection pending")

    # 5a: KeyboardInterrupt raised from a node (sync)
    def ki_graph():
        g = StateGraph(S)
        g.add_node("n1", _n("n1"))
        g.add_node("good", _n("good"))
        g.add_node("bad", _n("bad", sleep=0.25, raises=KeyboardInterrupt()))
        g.add_node("join", _n("join"))
        g.add_edge(START, "n1")
        for n in ("good", "bad"):
            g.add_edge("n1", n)
            g.add_edge(n, "join")
        g.add_edge("join", END)
        return g.compile

    for dur in ("sync", "async"):
        saver, exc = run_sync(
            case=f"s5-keyboardinterrupt-{dur}",
            durability=dur,
            trap="put_writes",
            predicate=writes_of("good"),
            graph_factory=ki_graph(),
        )
        classify(f"s5-keyboardinterrupt-{dur}", dur, saver, exc)

    # 5b: asyncio cancel of the whole run
    for dur in ("sync", "async"):
        saver, exc = run_async(
            case=f"s5-asyncio-cancel-{dur}",
            durability=dur,
            trap="put_writes",
            predicate=writes_of("n1"),
            mode="cancel",
            cancel_after=0.05,
        )
        classify(f"s5-asyncio-cancel-{dur}", dur, saver, exc)

    # 5c: GraphDrained via RunControl.request_drain()
    def drain_graph():
        g = StateGraph(S)

        def n1(state: S, runtime: Runtime) -> S:
            SIDE.append("n1")
            if runtime.control is not None:
                runtime.control.request_drain("sigterm")
            return {"log": ["n1"]}

        g.add_node("n1", n1)
        g.add_node("n2", _n("n2"))
        g.add_edge(START, "n1")
        g.add_edge("n1", "n2")
        g.add_edge("n2", END)
        return g.compile

    for dur in ("sync", "async"):
        SIDE.clear()
        clear_evidence()
        case = f"s5-graphdrained-{dur}"
        p("")
        p(f"--- CASE {case}  (durability={dur}, trap=put_writes)")
        exc = None
        with TrapSaver.from_conn_string(CONN) as saver:
            saver.arm(trap="put_writes", predicate=writes_of("n1"))
            graph = drain_graph()(checkpointer=saver)
            try:
                with alarm(60, case):
                    graph.invoke(
                        {"log": ["in"]},
                        {"configurable": {"thread_id": f"{RUN}-{case}"}},
                        durability=dur,
                        control=RunControl(),
                    )
            except BaseException as e:  # noqa: BLE001
                exc = e
            for line in saver.calls:
                p(f"    {line}")
        p(f"  OBSERVED: GraphDrained seen = {isinstance(exc, GraphDrained)}")
        classify(case, dur, saver, exc, note="drain requested")

    # ---------------------------------------------------------------- S6 ----
    hdr("S6. ABANDONED GENERATOR (GeneratorExit at the yield inside `with loop`)")
    p("A caller that breaks out of `for chunk in graph.stream(...)` throws")
    p("GeneratorExit into the generator, so the loop exits with exc_type != None.")
    for dur in ("sync", "async"):
        saver, exc = run_sync(
            case=f"s6-abandon-sync-{dur}",
            durability=dur,
            trap="put_writes",
            predicate=writes_of("n1"),
            mode="abandon",
            abandon_after=1,
        )
        classify(f"s6-abandon-sync-{dur}", dur, saver, exc, note="stream abandoned")
    for dur in ("sync", "async"):
        saver, exc = run_async(
            case=f"s6-abandon-async-{dur}",
            durability=dur,
            trap="put_writes",
            predicate=writes_of("n1"),
            mode="abandon",
            abandon_after=1,
        )
        classify(f"s6-abandon-async-{dur}", dur, saver, exc, note="astream abandoned")

    # ---------------------------------------------------------------- S7 ----
    hdr("S7. _error_handler_write_futs -- the concurrent.futures.wait path")
    p("A node with error_handler=... makes commit() append ERROR_SOURCE_NODE writes")
    p("and collect their futures; schedule_error_handler then WAITS on them")
    p("(concurrent.futures.wait, which never raises) before running the handler.")

    def eh_graph(async_=False, reject: str = "error"):
        g = StateGraph(S)

        def failing(state: S) -> S:
            SIDE.append("failing")
            raise NodeBoom("handled failure")

        def handler(state: S) -> S:
            SIDE.append("handler")
            return {"log": ["handled"]}

        g.add_node("failing", failing, error_handler=handler)
        g.add_node("after", _n("after"))
        g.add_edge(START, "failing")
        g.add_edge("failing", "after")
        g.add_edge("after", END)
        return g.compile

    def reject_error_write(writes=None, **kw):
        return writes is not None and any(
            c.startswith("__error") or "error" in c for c, _ in writes
        )

    for dur in ("sync", "async"):
        saver, exc = run_sync(
            case=f"s7-errhandler-{dur}",
            durability=dur,
            trap="put_writes",
            predicate=reject_error_write,
            graph_factory=eh_graph(),
        )
        classify(f"s7-errhandler-{dur}", dur, saver, exc, note="error-handler node ran")

    # ---------------------------------------------------------------- S8 ----
    hdr("S8. SUBGRAPH: fence fires inside a subgraph's checkpoint_ns")

    def subgraph_factory():
        inner = StateGraph(S)
        inner.add_node("i1", _n("i1"))
        inner.add_node("i2", _n("i2"))
        inner.add_edge(START, "i1")
        inner.add_edge("i1", "i2")
        inner.add_edge("i2", END)
        sub = inner.compile()
        outer = StateGraph(S)
        outer.add_node("o1", _n("o1"))
        outer.add_node("sub", sub)
        outer.add_node("o2", _n("o2"))
        outer.add_edge(START, "o1")
        outer.add_edge("o1", "sub")
        outer.add_edge("sub", "o2")
        outer.add_edge("o2", END)
        return outer.compile

    for dur in ("sync", "async", "exit"):
        for trap in ("put_writes", "put"):
            saver, exc = run_sync(
                case=f"s8-subgraph-{dur}-{trap}",
                durability=dur,
                trap=trap,
                predicate=ns_nonempty,
                graph_factory=subgraph_factory(),
            )
            classify(
                f"s8-subgraph-{dur}-{trap}", dur, saver, exc, note="rejected only inside subgraph ns"
            )

    # S8b: subgraph rejection racing an outer-node error is the compound case
    def subgraph_race_factory():
        inner = StateGraph(S)
        inner.add_node("i1", _n("i1"))
        inner.add_edge(START, "i1")
        inner.add_edge("i1", END)
        sub = inner.compile()
        outer = StateGraph(S)
        outer.add_node("o1", _n("o1"))
        outer.add_node("sub", sub)
        outer.add_node("bad", _n("bad", sleep=0.35, raises=NodeBoom("outer node blew up")))
        outer.add_node("join", _n("join"))
        outer.add_edge(START, "o1")
        outer.add_edge("o1", "sub")
        outer.add_edge("o1", "bad")
        outer.add_edge("sub", "join")
        outer.add_edge("bad", "join")
        outer.add_edge("join", END)
        return outer.compile

    for dur in ("sync", "async"):
        saver, exc = run_sync(
            case=f"s8b-subgraph-race-{dur}",
            durability=dur,
            trap="put_writes",
            predicate=ns_nonempty,
            graph_factory=subgraph_race_factory(),
        )
        classify(f"s8b-subgraph-race-{dur}", dur, saver, exc, note="subgraph fence + outer error")

    # --------------------------------------------------------------- S10 ----
    hdr("S10. REAL mid-run cancellation (the S5 cancel above finished too fast)")
    p("Slow trap (sleeps before raising) + slow node, so the rejection is either")
    p("IN FLIGHT or already stored in a pending future when the cancel lands.")

    async def cancel_case(case: str, dur: str, trap_delay: float, cancel_at: float):
        SIDE.clear()
        clear_evidence()
        asyncio.get_running_loop().set_exception_handler(_aio_handler)
        p("")
        p(f"--- CASE {case} (durability={dur}, trap_delay={trap_delay}, cancel_at={cancel_at})")
        g = StateGraph(S)
        g.add_node("n1", _an("n1"))
        g.add_node("n2", _an("n2", sleep=2.0))
        g.add_node("n3", _an("n3"))
        g.add_edge(START, "n1")
        g.add_edge("n1", "n2")
        g.add_edge("n2", "n3")
        g.add_edge("n3", END)
        exc = None
        async with AsyncTrapSaver.from_conn_string(CONN) as saver:
            saver.arm(trap="put_writes", predicate=writes_of("n1"), delay=trap_delay)
            graph = g.compile(checkpointer=saver)
            t = asyncio.ensure_future(
                graph.ainvoke(
                    {"log": ["in"]},
                    {"configurable": {"thread_id": f"{RUN}-{case}"}},
                    durability=dur,
                )
            )
            await asyncio.sleep(cancel_at)
            t.cancel()
            try:
                await t
            except BaseException as e:  # noqa: BLE001
                exc = e
            for line in saver.calls:
                p(f"    {line}")
            p(f"  OBSERVED: task.cancelled() = {t.cancelled()}")
        await asyncio.sleep(0.3)
        gc.collect()
        await asyncio.sleep(0.1)
        classify(case, dur, saver, exc, note=f"cancel_at={cancel_at}")

    for dur in ("sync", "async"):
        # rejection already raised and sitting in a pending future
        asyncio.run(cancel_case(f"s10-cancel-after-reject-{dur}", dur, 0.0, 0.5))
        # rejection still executing inside the trap when the cancel lands
        asyncio.run(cancel_case(f"s10-cancel-during-reject-{dur}", dur, 1.0, 0.3))

    # --------------------------------------------------------------- S11 ----
    hdr("S11. SIGINT delivered from OUTSIDE mid-run (realistic Ctrl-C / shutdown)")
    import os

    for dur in ("sync", "async"):
        case = f"s11-sigint-{dur}"
        SIDE.clear()
        clear_evidence()
        p("")
        p(f"--- CASE {case} (durability={dur}, trap=put_writes on n1)")
        g = StateGraph(S)
        g.add_node("n1", _n("n1"))
        g.add_node("n2", _n("n2", sleep=2.0))
        g.add_node("n3", _n("n3"))
        g.add_edge(START, "n1")
        g.add_edge("n1", "n2")
        g.add_edge("n2", "n3")
        g.add_edge("n3", END)
        exc = None
        with TrapSaver.from_conn_string(CONN) as saver:
            saver.arm(trap="put_writes", predicate=writes_of("n1"))
            graph = g.compile(checkpointer=saver)
            timer = threading.Timer(0.6, lambda: os.kill(os.getpid(), signal.SIGINT))
            timer.start()
            try:
                with alarm(60, case):
                    graph.invoke(
                        {"log": ["in"]},
                        {"configurable": {"thread_id": f"{RUN}-{case}"}},
                        durability=dur,
                    )
            except BaseException as e:  # noqa: BLE001
                exc = e
            timer.cancel()
            for line in saver.calls:
                p(f"    {line}")
        classify(case, dur, saver, exc, note="external SIGINT during n2")

    # --------------------------------------------------------------- S12 ----
    hdr("S12. the ONE apparently-safe cell: durability='sync' + trap=put")
    p("In S1 that cell SURFACED, but only because `predicate=always` rejected the")
    p("very first put (step=-1) before any node ran. Re-test with the fence arming")
    p("only at the superstep that contains the failing node, which is the real race.")

    def put_from_step(k: int):
        def f(metadata=None, **kw):
            return metadata is not None and (metadata.get("step") or -99) >= k

        return f

    for dur in ("sync", "async"):
        saver, exc = run_sync(
            case=f"s12-put-armed-late-{dur}",
            durability=dur,
            trap="put",
            predicate=put_from_step(1),
            graph_factory=race_graph(),
        )
        classify(
            f"s12-put-armed-late-{dur}",
            dur,
            saver,
            exc,
            note="fence arms at the put of the FAILING superstep",
        )

    # --------------------------------------------------------------- S13 ----
    hdr("S13. is durability='sync' + fence-in-put safe against the OTHER drop sites?")
    p("S12 showed sync+put survives a competing node error (the put future is")
    p("awaited every tick). Now hit that same cell with abandonment and SIGINT.")

    def put_from_step2(k: int):
        def f(metadata=None, **kw):
            return metadata is not None and (metadata.get("step") or -99) >= k

        return f

    saver, exc = run_sync(
        case="s13-syncput-abandoned-stream",
        durability="sync",
        trap="put",
        predicate=put_from_step2(1),
        mode="abandon",
        abandon_after=2,
    )
    classify("s13-syncput-abandoned-stream", "sync", saver, exc, note="caller broke out of stream")

    import os as _os

    case = "s13-syncput-sigint"
    SIDE.clear()
    clear_evidence()
    p("")
    p(f"--- CASE {case} (durability=sync, trap=put armed from step 1)")
    g = StateGraph(S)
    g.add_node("n1", _n("n1"))
    g.add_node("n2", _n("n2", sleep=2.0))
    g.add_node("n3", _n("n3"))
    g.add_edge(START, "n1")
    g.add_edge("n1", "n2")
    g.add_edge("n2", "n3")
    g.add_edge("n3", END)
    exc = None
    with TrapSaver.from_conn_string(CONN) as saver:
        # reject the put that lands while n2 is still running: arm on step>=2
        saver.arm(trap="put", predicate=put_from_step2(2), delay=1.0)
        graph = g.compile(checkpointer=saver)
        timer = threading.Timer(0.5, lambda: _os.kill(_os.getpid(), signal.SIGINT))
        timer.start()
        try:
            with alarm(60, case):
                graph.invoke(
                    {"log": ["in"]},
                    {"configurable": {"thread_id": f"{RUN}-{case}"}},
                    durability="sync",
                )
        except BaseException as e:  # noqa: BLE001
            exc = e
        timer.cancel()
        for line in saver.calls:
            p(f"    {line}")
    classify(case, "sync", saver, exc, note="SIGINT while the fenced put is in flight")

    # ---------------------------------------------------------------- S9 ----
    hdr("S9. SUMMARY")
    p(f"{'case':34} {'dur':6} {'rej':4} {'caller exc':22} {'surf':5} verdict")
    for case, dur, n, caller, surf, verdict in RESULTS:
        p(f"{case:34} {dur:6} {n:<4} {caller:22} {str(surf):5} {verdict}")
    silent = [r for r in RESULTS if r[5] == "DROPPED-SILENT"]
    p("")
    p(f"OBSERVED: configurations where the rejection was NEITHER surfaced NOR")
    p(f"          recorded anywhere observable: {len(silent)} / {len(RESULTS)}")
    for r in silent:
        p(f"          - {r[0]} (caller saw {r[3]})")
    never = [r for r in RESULTS if r[5] == "FENCE-NEVER-CONSULTED"]
    p(f"OBSERVED: configurations where the fence was never even consulted: {len(never)}")
    for r in never:
        p(f"          - {r[0]}")
    p("")
    p("DONE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        p("SCRIPT-LEVEL FAILURE:")
        p(traceback.format_exc())
        raise
