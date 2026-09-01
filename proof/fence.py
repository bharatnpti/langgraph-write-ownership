"""The fence: one table, one claim statement, one SQL guard. Nothing else.

Per-WRITE fencing (the Q4 seam), not a whole-run transaction (Q2): each
super-step must commit incrementally or a crashed worker leaves its successor
nothing to resume from. README explains every constraint encoded here.
"""
from __future__ import annotations

import sys

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver

CHECKPOINT_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes",
                     "checkpoint_migrations")

# The fence row MUST live in the same database as the checkpoint tables: the
# atomicity comes from sharing the write's implicit transaction.
FENCE_DDL = """create table if not exists proof_fence (
    thread_id text primary key, owner text not null, fence bigint not null)"""

# ONE statement whose WHERE tests exactly the column it writes (`fence`).
CLAIM_SQL = """update proof_fence set fence = fence + 1, owner = %s
 where thread_id = %s and fence = %s returning fence"""

# Fails SERVER-side. A client-side raise is not equivalent: measured here, a
# Python raise inside an open pipeline block still commits the already-queued
# write on BOTH pipeline branches (psycopg's Pipeline.__exit__ enqueues a Sync
# in _exit_gen regardless of the exception) -- only the supports_pipeline=False
# conn.transaction() fallback rolls it back. Divisor is a subquery so Postgres
# cannot constant-fold it into an unconditional error; the verdict is the
# error, never a rowcount.
GUARD_SQL = ("select 1 / (select count(*)::int from proof_fence"
             " where thread_id = %s and fence = %s)")


def setup_database(conninfo: str) -> None:
    """Idempotent, on a SEPARATE autocommit connection, once, not concurrently:
    MIGRATIONS[4..6] are CREATE INDEX CONCURRENTLY."""
    with psycopg.connect(conninfo, autocommit=True) as c:
        for t in (*CHECKPOINT_TABLES, "proof_fence"):
            c.execute(f"drop table if exists {t} cascade")
        PostgresSaver(c).setup()
        c.execute(FENCE_DDL)


def install_fence(conninfo: str, thread_id: str, owner: str) -> None:
    with psycopg.connect(conninfo, autocommit=True) as c:
        c.execute("insert into proof_fence values (%s,%s,1) on conflict"
                  " (thread_id) do update set owner=%s, fence=1",
                  (thread_id, owner, owner))


def read_fence(conninfo: str, thread_id: str) -> int | None:
    with psycopg.connect(conninfo, autocommit=True) as c:
        row = c.execute("select fence from proof_fence where thread_id=%s",
                        (thread_id,)).fetchone()
        return None if row is None else row[0]


def claim(conninfo: str, thread_id: str, expected: int, owner: str) -> int | None:
    """Take over the run. Read via RETURNING + fetchall(), never bare rowcount."""
    with psycopg.connect(conninfo, autocommit=True) as c:
        rows = c.execute(CLAIM_SQL, (owner, thread_id, expected)).fetchall()
        return rows[0][0] if rows else None


class FencedPostgresSaver(PostgresSaver):
    """The fence at the single seam covering every write of THREAD state:
    `_cursor()` is passed pipeline=True at exactly three sites -- put (:321),
    put_writes (:368), delete_thread (:390).

    Of the five OTHER `_cursor()` calls, four are reads (:159 list, :237
    get_tuple, :490 and :545 get_delta_channel_history) and one is not:
    setup() (:92) uses an unguarded `_cursor()` and writes -- 22 statements,
    20 of them DDL migrations plus `INSERT INTO checkpoint_migrations`. That is
    schema, not thread state, and it runs once before any worker starts, so
    leaving it unfenced is deliberate; but the seam is "every write of thread
    state", not "every write".
    """

    def __init__(self, conn, thread_id: str, fence: int, **kw) -> None:
        super().__init__(conn, **kw)
        self.fence_thread_id = thread_id
        self.fence_value = fence

    def _cursor(self, *, pipeline: bool = False):
        outer = super()._cursor(pipeline=pipeline)
        saver = self

        class _Guarded:
            def __enter__(self):
                self.cur = outer.__enter__()
                if pipeline:
                    try:
                        self.cur.execute(
                            GUARD_SQL, (saver.fence_thread_id, saver.fence_value))
                    except BaseException:
                        # Unwind before re-raising, or PostgresSaver.lock is
                        # never released and the next write blocks forever.
                        try:
                            outer.__exit__(*sys.exc_info())
                        except BaseException:
                            pass
                        raise
                return self.cur

            def __exit__(self, *exc):
                return outer.__exit__(*exc)

        return _Guarded()


def orphans(messages) -> list[str]:
    """Message-list well-formedness as the provider APIs define it: every
    tool_result needs a preceding matching tool_use, and every tool_use must be
    answered. OUR check, standing in for a provider's 400."""
    problems, offered, answered = [], set(), set()
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            offered.add(tc["id"])
        tcid = getattr(m, "tool_call_id", None)
        if tcid is not None:
            answered.add(tcid)
            if tcid not in offered:
                problems.append(f"tool_result tool_call_id={tcid!r} has no matching"
                                " tool_use anywhere before it in this message list")
    problems += [f"tool_use id={t!r} was never answered by a tool_result"
                 for t in sorted(offered - answered)]
    return problems
