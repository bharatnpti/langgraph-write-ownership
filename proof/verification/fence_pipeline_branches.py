"""Does the fence hold in the ONE _cursor() branch the proof never exercises?

PostgresSaver._cursor has three branches:
  1. self.pipe is not None            -> `finally: self.pipe.sync()`   NOT TESTED by proof.py
  2. conn.pipeline()                  -> tested (pipeline path)
  3. conn.transaction()               -> tested (supports_pipeline=False path)

Branch 1 is where the README's constraint "a Python raise inside a pipeline block
still commits the queued writes (psycopg enqueues a Sync in a finally)" would bite.
Measure both the SQL guard and a client-side raise there.

Provenance: copied verbatim from the adversarial-review scratchpad as
`branch1.py`. Only the sys.path setup below was changed, to point at this
repo's real spike/ and proof/ instead of the scratchpad's throwaway
`pristine/` copy of fence.py -- see ../README.md for the evidence-to-claim
mapping.
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent  # verification -> proof -> repo root
sys.path.insert(0, str(REPO / "spike"))
sys.path.insert(0, str(REPO / "proof"))

import psycopg  # noqa: E402
from _harness import pg_conninfo  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402

from fence import FencedPostgresSaver, install_fence, setup_database  # noqa: E402

CONNINFO = pg_conninfo("da_branch1")


def rows(tid: str) -> int:
    with psycopg.connect(CONNINFO, autocommit=True) as c:
        return c.execute("select count(*) from checkpoint_writes where thread_id=%s",
                         (tid,)).fetchone()[0]


def cfg(tid: str) -> dict:
    return {"configurable": {"thread_id": tid, "checkpoint_ns": "",
                             "checkpoint_id": "ckpt-1"}}


def attempt(tid: str, make_saver, label: str, conn_pipeline: bool) -> None:
    install_fence(CONNINFO, tid, "A")           # fence = 1
    with psycopg.connect(CONNINFO, autocommit=True) as c:   # someone else claims it
        c.execute("update proof_fence set fence=2 where thread_id=%s", (tid,))
    before = rows(tid)
    conn = psycopg.connect(CONNINFO, autocommit=True)
    err = "no exception"

    def body(pipe):
        nonlocal err
        saver = make_saver(conn, pipe, tid)
        try:
            saver.put_writes(cfg(tid), [("messages", [AIMessage("zombie write")])],
                             "task-zombie")
        except BaseException as e:  # noqa: BLE001
            err = (f"{type(e).__name__}: {str(e)[:52]}"
                   f" sqlstate={getattr(e, 'sqlstate', None)}")

    if conn_pipeline:
        with conn.pipeline() as pipe:
            body(pipe)
    else:
        body(None)
    conn.close()
    after = rows(tid)
    verdict = ("REFUSED (no row landed)" if after == before
               else "!!! WRITE COMMITTED ANYWAY")
    print(f"  {label:<44} raise={err:<62} rows {before}->{after}  {verdict}")


class ClientRaiseSaver(PostgresSaver):
    """The naive implementation: read the fence, raise in Python if stale, with the
    write already queued into the pipeline."""

    def __init__(self, conn, thread_id, fence, **kw):
        super().__init__(conn, **kw)
        self.t, self.f = thread_id, fence

    def _cursor(self, *, pipeline: bool = False):
        outer = super()._cursor(pipeline=pipeline)
        me = self

        class _Cur:
            def __init__(self, cur):
                self._c = cur

            def __getattr__(self, n):
                return getattr(self._c, n)

            def _check(self):
                with psycopg.connect(CONNINFO, autocommit=True) as c:
                    n = c.execute("select count(*) from proof_fence where"
                                  " thread_id=%s and fence=%s", (me.t, me.f)).fetchone()[0]
                if n != 1:
                    raise RuntimeError("fence stale (client-side raise after queue)")

            def execute(self, *a, **kw):
                r = self._c.execute(*a, **kw)
                self._check()
                return r

            def executemany(self, *a, **kw):
                r = self._c.executemany(*a, **kw)
                self._check()
                return r

        class _G:
            def __enter__(self):
                self.cur = outer.__enter__()
                return _Cur(self.cur) if pipeline else self.cur

            def __exit__(self, *e):
                return outer.__exit__(*e)

        return _G()


def main() -> None:
    setup_database(CONNINFO)
    print("BRANCH 1 (saver holds a connection-level Pipeline: self.pipe is not None)")
    attempt("b1-sql", lambda conn, pipe, tid:
            FencedPostgresSaver(conn, tid, 1, pipe=pipe),
            "proof's SQL guard (GUARD_SQL)", True)
    attempt("b1-py", lambda conn, pipe, tid:
            ClientRaiseSaver(conn, tid, 1, pipe=pipe),
            "naive client-side raise after queueing", True)
    attempt("b1-ctl", lambda conn, pipe, tid: PostgresSaver(conn, pipe=pipe),
            "no fence at all (control)", True)

    print("\nBRANCH 2 for contrast (self.pipe is None -> conn.pipeline() per write)")
    attempt("b2-sql", lambda conn, pipe, tid:
            FencedPostgresSaver(conn, tid, 1),
            "proof's SQL guard (GUARD_SQL)", False)
    attempt("b2-py", lambda conn, pipe, tid:
            ClientRaiseSaver(conn, tid, 1),
            "naive client-side raise after queueing", False)

    print("\nUNFENCED control (no guard at all) -- proves the write would land")
    attempt("ctl", lambda conn, pipe, tid: PostgresSaver(conn),
            "plain PostgresSaver, no fence", False)


if __name__ == "__main__":
    main()
