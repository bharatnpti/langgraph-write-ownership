# langgraph-write-ownership

Two things, both about one gap: **`langgraph-checkpoint-conformance` has no clause about
concurrent writers**, so a checkpointer that accepts writes from any number of workers on one
`thread_id` reports `conformance_level = FULL`.

1. **A reproduction** of the failure that silence permits — two OS processes, one thread, a real
   PostgreSQL, and a forked checkpoint chain that nothing reports.
2. **A proposed extended capability**, `write_ownership`, for the conformance suite: 15 clauses,
   purely additive, with patches that apply cleanly to `main` and to the published 0.0.2.

Discussion: <https://forum.langchain.com/c/oss-product-help-lc-and-lg/langgraph/13>

---

## Run the reproduction

```bash
cd spike
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install \
  langgraph langgraph-checkpoint-postgres deepagents "psycopg[binary,pool]" pgserver
cd ..
spike/.venv/bin/python proof/proof.py
```

About 25 seconds. **No API key and no Docker** — the model is a scripted `BaseChatModel` that
replays fixed messages, and PostgreSQL comes from `pgserver`, which runs a private cluster over a
unix socket. Exit code is 0 on `PROVEN`.

Recorded output: [`proof/proof.out`](proof/proof.out). Full write-up, including the two-tier
scoping: [`proof/README.md`](proof/README.md).

### What it shows

Worker A is `SIGSTOP`ed inside a model call. Worker B takes the thread over with
`invoke(None, config)` and finishes. Worker A is thawed and writes.

Both workers ran the same task from the same checkpoint, so both call `put_writes` with the same
deterministic `task_id`, and `INSERT … ON CONFLICT DO NOTHING` makes that first-writer-wins: A's
write is discarded while A is told it succeeded. A then writes its own child of that checkpoint, the
chain forks, and A's branch becomes the tip. B's terminal checkpoint is unreachable, and the answer
B already returned to its caller no longer matches the thread. Both processes exit 0. Nothing
raises and nothing is logged.

25 assertions. The chain fork is general to LangGraph plus `PostgresSaver`; a malformed message
list that one provider rejected with HTTP 400 is specific to `deepagents`' `DeltaChannel` and is
labelled as such throughout. A plain `MessagesState` graph forks identically but leaves each branch
internally consistent.

**Not proven:** one host, two processes, `SIGSTOP`, embedded PostgreSQL. No real multi-machine
network partition.

## The proposed capability

[`upstream/checkpoint-conformance-write-ownership/`](upstream/checkpoint-conformance-write-ownership/README.md)

A saver that advertises write ownership must **observably** refuse a write from a superseded owner.
Extended rather than base, because `tests/test_validate_memory.py` asserts `passed_all_base()`
against `InMemorySaver`; opt-in, so an unfenced deployment behaves exactly as it does today.

Additive with no base-class change: `_is_overridden` in `conformance/capabilities.py` returns
`True` for a method `BaseCheckpointSaver` does not define, so a capability is detected purely
because a saver defines it.

Three new files plus 21 lines of registration across four existing ones. Two patches, both verified
to apply cleanly — one against `main`, one against the sdist published to PyPI as 0.0.2 — plus an
idempotent `apply.py` so it survives a rebase.

```bash
spike/.venv/bin/python upstream/checkpoint-conformance-write-ownership/verdict.py
```

| Saver | Result |
|---|---|
| reference implementation, ~40 lines over `InMemorySaver` | **PASS** 15/15 |
| a saver that ignores supersession | **FAIL** — 7 pass, 8 fail |
| a saver that refuses **silently** | **FAIL** — 8 fail |
| a saver that refuses, reporting at a barrier only | **PASS** 15/15 |
| `InMemorySaver`, unmodified | capability not detected, base intact, still `FULL` |

That is the half an upstream reviewer can reproduce from this repository alone. `verdict.py` has a
second half that runs a fenced PostgreSQL saver from a separate project; it is not needed for any
claim here and **skips cleanly** when that project is absent.

## Supporting measurements

| | |
|---|---|
| [`spike/q_conformance.py`](spike/q_conformance.py) → [`.out`](spike/q_conformance.out) | The suite against `InMemorySaver`, `AsyncPostgresSaver` 3.1.2, and a deliberately fence-hostile saver. All three report `passed_all_base() = True` and `conformance_level = FULL` |
| [`spike/q3_verify1_swallow_paths.py`](spike/q3_verify1_swallow_paths.py) → [`.out`](spike/q3_verify1_swallow_paths.out) | Why raising from a saver is not a reliable refusal: of 56 configurations, **28** discarded the rejection with nothing recorded anywhere observable |
| [`proof/verification/`](proof/verification/README.md) | The adversarial re-tests and mutation pass behind the reproduction's claims |

`q_conformance.py` needs the upstream suite, which is deliberately not vendored here; it prints the
two `uv` commands that produce a copy if `CONFORMANCE_PKG` is unset.

## Pinned versions

`langgraph` 1.2.11 · `langgraph-checkpoint` 4.2.0 · `langgraph-checkpoint-postgres` 3.1.2 ·
`langgraph-checkpoint-conformance` 0.0.2 · `deepagents` 0.7.11 · `psycopg` 3.3.4 ·
PostgreSQL 16.2

Upstream state was verified against `main` at `11ee185` (2026-08-28).

## Licence

MIT — see [`LICENSE`](LICENSE).
