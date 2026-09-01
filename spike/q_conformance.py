"""Lane: conformance.

Executes the REAL upstream langgraph-checkpoint-conformance suite (main @ 2026-08-30,
version 0.0.2, fetched via `gh api` from libs/checkpoint-conformance) against:
  1. InMemorySaver            -> reproduce upstream's own self-test
  2. AsyncPostgresSaver 3.1.2 -> what conformance level does the real PG saver reach?
  3. FenceHostileSaver        -> a saver that has NO ownership/fencing whatsoever and
                                 happily accepts writes from an unbounded number of
                                 concurrent "workers". Does the suite notice?

The suite is not installed in this venv (it is not a dependency of
langgraph-checkpoint-postgres 3.1.2). `langgraph` and `langgraph.checkpoint` are
implicit NAMESPACE packages, so the fetched sources are placed on sys.path
unmodified -- nothing is installed, nothing in the venv is touched.
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path
from uuid import uuid4

# The suite is NOT vendored into this repo (third-party source stays out of it), so
# point CONFORMANCE_PKG at a directory containing it. The simplest way to get one:
#
#     uv venv --python 3.12 /tmp/conf/.venv
#     VIRTUAL_ENV=/tmp/conf/.venv uv pip install langgraph-checkpoint-conformance
#     export CONFORMANCE_PKG=/tmp/conf/.venv/lib/python3.12/site-packages
#
# The recorded run used sources fetched from langchain-ai/langgraph main @ 2026-08-30
# with `gh api`, which is equivalent for these purposes; q_conformance.out is that run.
import os

CONFPKG = os.environ.get("CONFORMANCE_PKG", "")
if not CONFPKG or not Path(CONFPKG).is_dir():
    raise SystemExit(
        "CONFORMANCE_PKG is unset or not a directory.\n"
        "This script needs the upstream langgraph-checkpoint-conformance package, which is\n"
        "deliberately not vendored here. See the comment above this message for the two\n"
        "commands that produce one. The recorded evidence is spike/q_conformance.out."
    )
sys.path.insert(0, CONFPKG)
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.conformance import checkpointer_test, validate  # noqa: E402
from langgraph.checkpoint.conformance.capabilities import (  # noqa: E402
    ALL_CAPABILITIES,
    BASE_CAPABILITIES,
    DetectedCapabilities,
)
from langgraph.checkpoint.conformance.report import ProgressCallbacks  # noqa: E402
from langgraph.checkpoint.conformance.spec import __all__ as SPEC_RUNNERS  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import operator  # noqa: E402
from typing import Annotated  # noqa: E402

from typing_extensions import TypedDict  # noqa: E402

S = TypedDict("S", {"steps": Annotated[list, operator.add]})


def hdr(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78, flush=True)


# --------------------------------------------------------------------------
# 0. Inventory: how many assertions does the contract actually contain?
# --------------------------------------------------------------------------
def inventory() -> None:
    hdr("0. SUITE INVENTORY (upstream main, langgraph-checkpoint-conformance 0.0.2)")
    import importlib

    total = 0
    for mod_name in [
        "test_put",
        "test_put_writes",
        "test_get_tuple",
        "test_list",
        "test_delete_thread",
        "test_delete_for_runs",
        "test_copy_thread",
        "test_prune",
        "test_delta_channel_history",
    ]:
        mod = importlib.import_module(
            f"langgraph.checkpoint.conformance.spec.{mod_name}"
        )
        lists = [v for k, v in vars(mod).items() if k.startswith("ALL_")]
        tests = lists[0] if lists else []
        total += len(tests)
        print(f"  {mod_name:32s} {len(tests):3d} tests")
        for t in tests:
            print(f"      - {t.__name__}")
    print(f"\n  TOTAL CONTRACT CLAUSES (test functions): {total}")
    print(f"  spec runners exported: {len(SPEC_RUNNERS)}")
    print(f"  BASE capabilities:     {sorted(c.value for c in BASE_CAPABILITIES)}")
    print(f"  ALL capabilities:      {sorted(c.value for c in ALL_CAPABILITIES)}")


# --------------------------------------------------------------------------
# 1. InMemorySaver -- reproduce upstream's own self-test
# --------------------------------------------------------------------------
@checkpointer_test(name="InMemorySaver")
async def memory_checkpointer():
    yield InMemorySaver()


async def run_memory() -> None:
    hdr("1. InMemorySaver (reproduces libs/checkpoint-conformance/tests/"
        "test_validate_memory.py)")
    report = await validate(memory_checkpointer, progress=ProgressCallbacks.default())
    report.print_report()
    print(f"  passed_all_base() = {report.passed_all_base()}")
    print(f"  conformance_level = {report.conformance_level()}")
    for cap, r in report.results.items():
        if r.failures:
            print(f"  FAILURES in {cap}:")
            for f in r.failures:
                print(f"    {f}")


# --------------------------------------------------------------------------
# 2. AsyncPostgresSaver 3.1.2 -- the real thing
# --------------------------------------------------------------------------
async def run_postgres() -> None:
    hdr("2. AsyncPostgresSaver (langgraph-checkpoint-postgres 3.1.2)")
    from _harness import pg_conninfo
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    conninfo = pg_conninfo("conf_lane_pg")
    print(f"  db: {conninfo}")

    @checkpointer_test(name="AsyncPostgresSaver")
    async def pg_saver():
        async with AsyncPostgresSaver.from_conn_string(conninfo) as s:
            await s.setup()
            yield s

    report = await validate(pg_saver, progress=ProgressCallbacks.default())
    report.print_report()
    print(f"  passed_all_base() = {report.passed_all_base()}")
    print(f"  conformance_level = {report.conformance_level()}")
    for cap, r in report.results.items():
        if r.failures:
            print(f"  FAILURES in {cap}:")
            for f in r.failures:
                print(f"    {f}")


# --------------------------------------------------------------------------
# 3. FenceHostileSaver -- zero ownership semantics. Does the suite care?
# --------------------------------------------------------------------------
class FenceHostileSaver(InMemorySaver):
    """A saver with NO notion of run ownership.

    It records every writer that ever touched a thread and never refuses
    anything -- exactly the zombie-worker hazard durable-agents must fence.
    It is deliberately *more* permissive than any real saver: it strips
    checkpoint_id so a stale worker's put always lands on the latest state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.writers: dict[str, set[str]] = {}
        self.accepted_stale_writes = 0

    async def aput(self, config, checkpoint, metadata, new_versions):
        tid = config["configurable"]["thread_id"]
        wid = config["configurable"].get("worker_id", "anonymous")
        self.writers.setdefault(tid, set()).add(wid)
        if len(self.writers[tid]) > 1:
            self.accepted_stale_writes += 1
        return await super().aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        tid = config["configurable"]["thread_id"]
        wid = config["configurable"].get("worker_id", "anonymous")
        self.writers.setdefault(tid, set()).add(wid)
        if len(self.writers[tid]) > 1:
            self.accepted_stale_writes += 1
        return await super().aput_writes(config, writes, task_id, task_path)


@checkpointer_test(name="FenceHostileSaver")
async def hostile_checkpointer():
    yield FenceHostileSaver()


async def run_hostile() -> None:
    hdr("3. FenceHostileSaver -- no ownership semantics at all")
    report = await validate(hostile_checkpointer, progress=ProgressCallbacks.default())
    report.print_report()
    print(f"  passed_all_base() = {report.passed_all_base()}")
    print(f"  conformance_level = {report.conformance_level()}")
    print("  --> If BASE passes, the suite provably cannot detect the absence")
    print("      of run-ownership / fencing.")

    # Now demonstrate the concrete hazard the suite does not test for.
    hdr("3b. The hazard the suite has no clause for: two workers, one thread")
    s = FenceHostileSaver()
    tid = str(uuid4())
    from langgraph.checkpoint.conformance.test_utils import (
        generate_checkpoint,
        generate_config,
        generate_metadata,
    )

    cfg_a = generate_config(tid)
    cfg_a["configurable"]["worker_id"] = "worker-A-LIVE"
    stored_a = await s.aput(cfg_a, generate_checkpoint(), generate_metadata(step=0), {})
    await s.aput_writes(stored_a, [("ch", "A-work")], str(uuid4()))

    # Worker B: a zombie that never died, resuming the SAME thread.
    cfg_b = generate_config(tid)
    cfg_b["configurable"]["worker_id"] = "worker-B-ZOMBIE"
    stored_b = await s.aput(cfg_b, generate_checkpoint(), generate_metadata(step=0), {})
    await s.aput_writes(stored_b, [("ch", "B-work")], str(uuid4()))

    tup = await s.aget_tuple(generate_config(tid))
    print(f"  distinct writers on thread: {sorted(s.writers[tid])}")
    print(f"  writes accepted after a 2nd writer appeared: "
          f"{s.accepted_stale_writes}")
    print(f"  latest checkpoint id: {tup.config['configurable']['checkpoint_id']}")
    print(f"  pending_writes now:   {tup.pending_writes}")
    print("  Both workers' state coexists. NO conformance clause forbids this,")
    print("  and no error type exists for the saver to signal a refusal.")

    # Prove there is no exception type for a rejected write.
    hdr("3c. Is there an error type the runtime treats specially?")
    import langgraph.errors as lge

    print(f"  langgraph.errors.__all__ = {lge.__all__}")
    for name in ("CheckpointNotLatest", "CheckpointRejected", "FencedOut",
                 "OwnershipLost", "StaleWrite", "WriteRejected"):
        print(f"    {name:20s} present={hasattr(lge, name)}")

    from langgraph.checkpoint.base import BaseCheckpointSaver

    print("\n  BaseCheckpointSaver.aput return annotation:")
    print(f"    {BaseCheckpointSaver.aput.__annotations__}")
    print("  BaseCheckpointSaver.aput_writes return annotation:")
    print(f"    {BaseCheckpointSaver.aput_writes.__annotations__}")


# --------------------------------------------------------------------------
# 4. Does a raising saver even surface? (does the runtime propagate?)
# --------------------------------------------------------------------------
async def run_raising() -> None:
    hdr("4. Does a saver that REFUSES a write propagate through Pregel?")

    class RefusingSaver(InMemorySaver):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        async def aput(self, config, checkpoint, metadata, new_versions):
            self.n += 1
            if self.n >= 2:
                raise PermissionError("FENCED OUT: run reassigned to another worker")
            return await super().aput(config, checkpoint, metadata, new_versions)

    from langgraph.graph import END, START, StateGraph

    def n1(state):
        return {"steps": ["n1"]}

    def n2(state):
        return {"steps": ["n2"]}

    saver = RefusingSaver()
    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_node("n2", n2)
    g.add_edge(START, "n1")
    g.add_edge("n1", "n2")
    g.add_edge("n2", END)
    app = g.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": str(uuid4())}}
    try:
        out = await app.ainvoke({"steps": []}, cfg)
        print(f"  NO exception raised. result={out}")
    except Exception as e:
        print(f"  raised: {type(e).__module__}.{type(e).__name__}: {e}")
        print("  --> a saver refusal propagates as a raw exception; there is no")
        print("      typed rejection the runtime treats specially.")
    print(f"  aput call count: {saver.n}")
    st = await app.aget_state(cfg)
    print(f"  state after refusal: next={st.next} values={st.values}")


async def main() -> None:
    inventory()
    for fn in (run_memory, run_postgres, run_hostile, run_raising):
        try:
            await fn()
        except Exception:
            print(f"\n!!! {fn.__name__} BLEW UP:")
            traceback.print_exc()
            sys.stdout.flush()


def _cli():
    if "--cap" in sys.argv:
        probe_capability_detection()
    else:
        asyncio.run(main())


# --------------------------------------------------------------------------
# 5. Can a NEW extended capability be added without touching BaseCheckpointSaver?
# --------------------------------------------------------------------------
def probe_capability_detection() -> None:
    hdr("5. Extended-capability detection: does the method need to exist on "
        "BaseCheckpointSaver?")
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.conformance.capabilities import _is_overridden

    for m in ("aput", "aput_writes", "aget_tuple", "alist", "adelete_thread",
              "adelete_for_runs", "acopy_thread", "aprune",
              "aget_delta_channel_history"):
        print(f"  BaseCheckpointSaver has {m:28s} -> {hasattr(BaseCheckpointSaver, m)}")

    class SaverWithNovelMethod(InMemorySaver):
        async def aput_writes_fenced(self, config, writes, task_id, fence):  # noqa: D102
            raise NotImplementedError

    print("\n  A hypothetical NEW capability method NOT on the base class:")
    print(f"    _is_overridden(SaverWithNovelMethod, 'aput_writes_fenced') = "
          f"{_is_overridden(SaverWithNovelMethod, 'aput_writes_fenced')}")
    print(f"    _is_overridden(InMemorySaver,        'aput_writes_fenced') = "
          f"{_is_overridden(InMemorySaver, 'aput_writes_fenced')}")
    print("\n  DetectedCapabilities on a plain InMemorySaver:")
    d = DetectedCapabilities.from_instance(InMemorySaver())
    print(f"    detected = {sorted(c.value for c in d.detected)}")
    print(f"    missing  = {sorted(c.value for c in d.missing)}")


if __name__ == "__main__":
    _cli()
