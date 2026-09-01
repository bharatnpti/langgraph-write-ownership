"""E9 -- does a client-side raise INSIDE the pipeline block still commit the
write that was already queued?  Tested at each of the three _cursor(pipeline=True)
branches, on a fresh key so nothing can mask the row via ON CONFLICT.

Provenance: copied VERBATIM (not one character changed) from the adversarial-
review scratchpad as `e9_queued_commit.py`. This is the mutation-6-shaped
probe: it is why "the sqlstate=='22012' assertion is load-bearing" in
../README.md -- see that file for the full evidence-to-claim mapping. No path
fix was needed: this script already hardcodes this repo's absolute path.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spike"))
sys.path.insert(0, str(ROOT / "proof"))

import psycopg  # noqa: E402
from _harness import pg_conninfo  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402

from fence import setup_database  # noqa: E402

CI = pg_conninfo("e9_queued")
setup_database(CI)


class RaiseAfterQueue(PostgresSaver):
    """m6-shaped mutant: queue the write, then raise in Python, client-side."""

    def _cursor(self, *, pipeline: bool = False):
        outer = super()._cursor(pipeline=pipeline)

        class _Cur:
            def __init__(self, cur):
                self._c = cur

            def __getattr__(self, n):
                return getattr(self._c, n)

            def execute(self, *a, **kw):
                r = self._c.execute(*a, **kw)
                raise RuntimeError("client-side raise after queue")

            def executemany(self, *a, **kw):
                self._c.executemany(*a, **kw)
                raise RuntimeError("client-side raise after queue")

        class _G:
            def __enter__(self):
                self.cur = outer.__enter__()
                return _Cur(self.cur) if pipeline else self.cur

            def __exit__(self, *exc):
                return outer.__exit__(*exc)

        return _G()


def rows(tid):
    with psycopg.connect(CI, autocommit=True) as c:
        return c.execute("select count(*) from checkpoint_writes where thread_id=%s",
                         (tid,)).fetchone()[0]


for label, mk in (("self.pipe set (connection-level Pipeline)", "pipe"),
                  ("supports_pipeline=True (conn.pipeline() per write)", "sp_true"),
                  ("supports_pipeline=False (conn.transaction() fallback)", "sp_false")):
    tid = f"e9-{mk}"
    cfg = {"configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": "cp1"}}
    exc = None
    conn = psycopg.connect(CI, autocommit=True, prepare_threshold=0,
                           row_factory=psycopg.rows.dict_row)
    try:
        if mk == "pipe":
            with conn.pipeline() as pipe:
                s = RaiseAfterQueue(conn, pipe=pipe)
                try:
                    s.put_writes(cfg, [("messages", "x")], "task-1")
                except BaseException as e:  # noqa: BLE001
                    exc = e
        else:
            s = RaiseAfterQueue(conn)
            s.supports_pipeline = (mk == "sp_true")
            try:
                s.put_writes(cfg, [("messages", "x")], "task-1")
            except BaseException as e:  # noqa: BLE001
                exc = e
    except BaseException as e:  # noqa: BLE001
        exc = exc or e
    finally:
        try:
            conn.close()
        except BaseException:
            pass
    n = rows(tid)
    print(f"  {label}")
    print(f"      raised {type(exc).__name__}: {str(exc)[:50]!r}")
    print(f"      checkpoint_writes rows for the thread AFTERWARDS = {n}"
          f"   -> queued write {'COMMITTED ANYWAY' if n else 'was NOT committed'}")
