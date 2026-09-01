#!/usr/bin/env python3
"""Run every WRITE_OWNERSHIP verdict and print them. Output: `verdict.out`.

    spike/.venv/bin/python upstream/checkpoint-conformance-write-ownership/verdict.py

Two halves, and the first is the one an upstream reviewer can reproduce without this
repository:

**In memory** -- five savers over `InMemorySaver`, no database, no network. Establishes
that the capability is implementable in about forty lines, that it fails a saver which
ignores supersession, that it also fails one which enforces silently, that it *passes* one
which reports only at the barrier, and that a plain saver is untouched.

**Against PostgreSQL** -- `durable-agents`' own fence, the published
`AsyncPostgresSaver` 3.1.2, and a mutant of ours with the fence removed but every line of
bookkeeping intact. That last one is the only way to say what the fence itself earns.

Skips the second half cleanly when `durable_agents` or the embedded cluster is not
available, so this is runnable from a bare checkout of the patch.
"""
from __future__ import annotations

import asyncio
import pathlib
import shutil
import sys
import tempfile
import uuid

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load_patched_suite():
    """Import the suite with this capability applied, without touching site-packages."""
    try:
        import langgraph.checkpoint.conformance as installed
    except ImportError:
        sys.exit(
            "langgraph-checkpoint-conformance is not installed.\n"
            "  uv pip install --no-deps langgraph-checkpoint-conformance"
        )
    root = pathlib.Path(tempfile.mkdtemp(prefix="wo-verdict-"))
    target = root / "langgraph" / "checkpoint"
    target.mkdir(parents=True)
    shutil.copytree(pathlib.Path(installed.__file__).parent, target / "conformance")
    shutil.copytree(
        HERE / "tree" / "langgraph" / "checkpoint" / "conformance",
        target / "conformance",
        dirs_exist_ok=True,
    )
    for name in [n for n in sys.modules if n.startswith("langgraph.checkpoint.conformance")]:
        del sys.modules[name]
    sys.path.insert(0, str(root))
    import langgraph.checkpoint.conformance as patched

    return patched


CAP = "write_ownership"


def _verdict(report) -> str:
    r = report.results[CAP]
    if not r.detected:
        return "NOT DETECTED"
    state = {True: "PASS", False: "FAIL", None: "SKIPPED"}[r.passed]
    return f"{state}  ({r.tests_passed} passed, {r.tests_failed} failed)"


def _failures(report) -> list[str]:
    return sorted({f.split(":", 1)[0] for f in report.results[CAP].failures})


def _report(label: str, report, *, expect: str) -> None:
    print(f"\n  {label}")
    print(f"    verdict:  {_verdict(report)}")
    print(f"    expected: {expect}")
    for name in _failures(report):
        print(f"      FAIL  {name}")


# ------------------------------------------------------------------- in memory ---


async def _in_memory(suite) -> None:
    sys.path.insert(0, str(HERE / "tree"))
    import tests.test_write_ownership as ref

    print("=" * 78)
    print("  HALF 1 -- in memory. No database, no network.")
    print("=" * 78)

    for reg, expect in (
        (ref.owned_checkpointer, "PASS -- the reference implementation"),
        (ref.hostile_checkpointer, "FAIL -- ignores supersession entirely"),
        (ref.silent_checkpointer, "FAIL -- refuses durably, never says so"),
        (ref.deferred_checkpointer, "PASS -- refuses, reports at the barrier only"),
    ):
        _report(reg.name, await suite.validate(reg, capabilities={CAP}), expect=expect)

    full = await suite.validate(ref.plain_checkpointer)
    print("\n  InMemorySaver (unmodified), full suite")
    print(f"    write_ownership:  {_verdict(full)}")
    print(f"    passed_all_base:  {full.passed_all_base()}")
    print(f"    conformance_level:{full.conformance_level()}")
    print("    expected:         NOT DETECTED, base intact -- extended, never base")


# ---------------------------------------------------------------- postgres ---


def _pg_conninfo() -> str | None:
    try:
        import pgserver
        import psycopg
    except ImportError:
        return None
    data = REPO / "spike" / "pgdata"
    if not data.exists():
        return None
    server = pgserver.get_server(str(data), cleanup_mode=None)
    name = "wo_verdict_" + uuid.uuid4().hex[:12]
    with psycopg.connect(server.get_uri(), autocommit=True) as c:
        c.execute(f'create database "{name}"')
    return server.get_uri().replace("/postgres?", f"/{name}?")


async def _postgres(suite) -> None:
    print()
    print("=" * 78)
    print("  HALF 2 -- against a real PostgreSQL 16.2.")
    print("=" * 78)

    sys.path.insert(0, str(REPO / "src"))
    try:
        from durable_agents import conformance as mod
        from durable_agents.langgraph.adapter import setup_database
        from durable_agents.store.postgres import PostgresRunStore
    except ImportError as exc:
        print(f"\n  SKIPPED: durable_agents is not importable ({exc})")
        return

    conninfo = _pg_conninfo()
    if conninfo is None:
        print("\n  SKIPPED: no pgserver, or spike/pgdata does not exist")
        return

    import psycopg
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    def prepared() -> str:
        ci = _pg_conninfo()
        store = PostgresRunStore(ci)
        setup_database(ci, store)
        store.close()
        return ci

    # ours
    _report(
        "durable-agents (FencedAsyncPostgresSaver + durable_runs)",
        await suite.validate(
            mod.register_conformance_checkpointer(prepared(), name="durable-agents"),
            capabilities={CAP},
        ),
        expect="PASS -- the fence refuses a superseded owner",
    )

    # the published saver, advertising only
    ci = prepared()

    class Advertising(AsyncPostgresSaver):
        async def aclaim_write_ownership(self, thread_id: str):
            return self

    async def factory():
        conn = await psycopg.AsyncConnection.connect(ci, autocommit=True)
        try:
            yield Advertising(conn)
        finally:
            await conn.close()

    _report(
        "AsyncPostgresSaver 3.1.2, unmodified (advertising only)",
        await suite.validate(
            suite.checkpointer_test(name="AsyncPostgresSaver")(factory),
            capabilities={CAP},
        ),
        expect="FAIL -- any worker may write any thread",
    )

    # ours, fence removed, bookkeeping intact
    class FenceRemoved(AsyncPostgresSaver):
        def __init__(self, conn, *, run_id: str, fence_token: int, **kw):
            super().__init__(conn, **kw)

    original = mod.FencedAsyncPostgresSaver
    mod.FencedAsyncPostgresSaver = FenceRemoved  # type: ignore[assignment]
    try:
        _report(
            "durable-agents with the FENCE REMOVED (bookkeeping intact)",
            await suite.validate(
                mod.register_conformance_checkpointer(prepared(), name="mutant"),
                capabilities={CAP},
            ),
            expect="FAIL -- and only on durability: this is what the fence earns",
        )
    finally:
        mod.FencedAsyncPostgresSaver = original  # type: ignore[assignment]


async def main() -> None:
    suite = _load_patched_suite()
    from langgraph.checkpoint.conformance.spec.test_write_ownership import (
        ALL_WRITE_OWNERSHIP_TESTS,
    )

    print(f"WRITE_OWNERSHIP -- {len(ALL_WRITE_OWNERSHIP_TESTS)} clauses\n")
    await _in_memory(suite)
    await _postgres(suite)
    print("\ndone.")


if __name__ == "__main__":
    asyncio.run(main())
