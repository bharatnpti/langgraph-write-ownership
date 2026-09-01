# The overwrite proof

Companion to [`README.md`](README.md) / [`proof.py`](proof.py), which prove the **dropped write**.
This proves its worse sibling: the **overwritten** write. Same two-process setup, same embedded
Postgres, same fence — a different `ON CONFLICT` clause, and a different kind of damage.

```bash
spike/.venv/bin/python proof/overwrite.py     # ~30s, exit 0 on PROVEN
```

19 assertions, 0 failures. Full stdout: [`overwrite.out`](overwrite.out). Idempotent — its own
database (`durable_agents_overwrite`), dropped and recreated every run. No API keys: the graph is
a plain `StateGraph`, no model at all.

## Why it is worse than the proven one

`put_writes` picks between two statements
(`langgraph/checkpoint/postgres/__init__.py:363-367`):

| batch | statement | who wins a collision | what is at stake |
|---|---|---|---|
| **any** channel outside `WRITES_IDX_MAP` | `INSERT … DO NOTHING` | the **first** writer | transcript — proven in `README.md` |
| **every** channel inside it | `UPSERT … DO UPDATE SET channel, type, blob` | the **second** writer | `__resume__`, `__interrupt__`, `__error__`, `__scheduled__` — **control data** |

Under `DO NOTHING` a stale worker *loses*: it is refused, silently, and the live worker's row
stands. Under `DO UPDATE` a stale worker *wins*: it replaces the live worker's row. And the rows
it can replace are not messages — they are the inputs the run will be re-executed on.

`README.md` recorded this as "an untested exposure", measured only at the SQL level, and guessed
it "would need a different scenario (a human-in-the-loop interrupt inside a tool)". That guess was
wrong in the reassuring direction: it needs **no tool and no `deepagents`** — a two-line plain
`StateGraph` reaches it.

## Three levels of evidence, and which claim rests on which

Every line of output is tagged. This matters more than the result.

| tag | means |
|---|---|
| `READ IN SOURCE` | quoted from the installed langgraph, with `file:line`. Argument, not measurement. |
| `HAND-DRIVEN SQL` | statements this file issued against real Postgres. Measurement, but not a graph run. |
| `REAL GRAPH RUN` | observed through `graph.invoke()`, the two contending workers in separate OS processes. |

The end-to-end claim is `REAL GRAPH RUN` only. Nothing in Part 2 or Part 3 calls `put_writes` by
hand.

## Pinned

Same cluster and pins as `README.md`: Python 3.12.13, `langgraph` 1.2.11,
`langgraph-checkpoint` 4.2.0, `langgraph-checkpoint-postgres` 3.1.2, `langchain-core` 1.6.1,
`psycopg` 3.3.4 (`has_pipeline()` True), PostgreSQL 16.2 embedded via `pgserver` 0.1.4.
`deepagents` is not involved. A verdict here says nothing about any other version.

## Part 1 — the mechanism, hand-driven

Both real statements, two writers, one conflict key:

```
INSERT .. ON CONFLICT DO NOTHING -> rows=1  blob='SURVIVOR'  task_path='path-of-survivor'
UPSERT .. ON CONFLICT DO UPDATE  -> rows=1  blob='ZOMBIE'    task_path='path-of-survivor'
```

Then the same collision **through `PostgresSaver.put_writes()` itself**, two savers on two
connections, so langgraph chooses the branch rather than this file asserting which one it would
choose:

```
ALL channels special  [('__resume__', v)]  -> idx=[-4]     __resume__='ZOMBIE'
MIXED  [('__resume__', v), ('log', [v])]   -> idx=[-4, 1]  __resume__='SURVIVOR'
```

One ordinary channel anywhere in the batch flips the whole batch back to first-writer-wins.

And the conflict key's last column is fixed per channel, which is why two workers cannot help
colliding:

```
t-special  idx=-4 __resume__   idx=-3 __interrupt__   idx=-2 __scheduled__   idx=-1 __error__
t-ordinary idx=0  log          idx=1  other
```

Two small artifacts, measured and worth knowing:

- **`task_path` is not in the `SET` list.** After a `DO UPDATE` the row carries the *second*
  writer's blob under the *first* writer's `task_path`. Since `idx` is derived from the channel
  for special writes, the reachable effect of the `SET` list is `blob`+`type` replacement at a
  fixed `(channel, idx)` — not a channel swap.
- **A single batch can self-collide.** Part 1b observes the real batch
  `['__resume__', '__resume__', 'log']`: `interrupt()` appends `(RESUME, scratchpad.resume)` on
  every call, both entries map to `idx=-4`, and `executemany` sends both. Harmless here — the two
  entries are the same list object and serialise identically — but note `_loop.py:419-421`
  de-duplicates special writes *last-wins* only when the batch is **all** special, while the SQL
  that receives this mixed batch is *first*-wins. The two disagree; nothing depends on it in this
  scenario.

## Part 1b — which batches a real run actually routes to `DO UPDATE`

One worker, one thread, `start → interrupt → resume → interrupt → resume → done`, with a tap on
`put_writes`:

```
cp ...a994f8cb  a93461c1     ['log', 'branch:to:approve']         -> INSERT/DO NOTHING
cp ...0b1c5501  39b80409     ['__interrupt__']                    -> UPSERT/DO UPDATE
cp ...0b1c5501  NULL_TASK_ID ['__resume__']                       -> UPSERT/DO UPDATE
cp ...0b1c5501  39b80409     ['__interrupt__', '__resume__']      -> UPSERT/DO UPDATE
cp ...0b1c5501  NULL_TASK_ID ['__resume__']                       -> UPSERT/DO UPDATE
cp ...0b1c5501  39b80409     ['__resume__', '__resume__', 'log']  -> INSERT/DO NOTHING
```

**4 of 6 batches take the `DO UPDATE` path.** The exposure is not a corner of the API; it is the
normal traffic of a human-in-the-loop pause. Three reachable shapes, read in source and matched to
the tap above:

| source | batch | task_id |
|---|---|---|
| `pregel/_loop.py:919-926` | `Command(resume=v)` → `[(RESUME, v)]` | `NULL_TASK_ID` |
| `pregel/_runner.py:585-591` | `GraphInterrupt` → `[(INTERRUPT, …)]` + any `RESUME` | the real task id |
| `pregel/_runner.py:596-604` | an unhandled node error before any channel write → `[(ERROR, exc)]` | the real task id |

The third is **not exercised** here. Note also that every checkpoint keeps *one* row per
`(task_id, __resume__)`, and that row is the accumulated list of human inputs already consumed by
that task — so it outlives the run that wrote it and is read by the **next** worker.

## Part 2 — the end-to-end overwrite, two OS processes

Graph: one node, two `interrupt()` calls, `log: Annotated[list[str], operator.add]`. A plain
`StateGraph` on purpose — `__resume__` is a LangGraph control channel, so nothing here depends on
how the transcript channel is declared, and none of `README.md`'s Tier 2 (`DeltaChannel`) applies.

| stage | what happens |
|---|---|
| 0 | a run reaches `interrupt()` and pauses. Tip `C`, step 0, one `__interrupt__` row. |
| 1 | **worker A** resumes with `'A-VALUE'`, loads `C`, and is **`SIGSTOP`'d** holding an unissued `__resume__` write for `C`. `ps` state `T` — frozen, not killed. |
| 2 | **worker B** — a separate OS process — resumes the same thread with `'B-VALUE'`, writes nothing else first, runs the node, hits the second `interrupt()`, pauses. |
| 3 | A is **`SIGCONT`'d** and allowed to write. |
| 4 | a **later worker** resumes the second interrupt with `'SECOND'`. The run completes. |

Stage 2 leaves the survivor's input as the durable control data at `C`:

```
NULL_TASK_ID   idx=-4  __resume__     value=B-VALUE
task f4c5e95e  idx=-4  __resume__     value=['B-VALUE']
task f4c5e95e  idx=-3  __interrupt__  value=[Interrupt(value='approve step 2?', ...)]
```

Stage 3 replaces it:

```
NULL_TASK_ID   idx=-4  __resume__     value=A-VALUE
task f4c5e95e  idx=-4  __resume__     value=['A-VALUE']
task f4c5e95e  idx=-3  __interrupt__  value=[Interrupt(value='approve step 2?', ...)]
```

The asserted verdict lines, verbatim:

```
[PASS] THE OVERWRITE: the stale worker's __resume__ REPLACED the survivor's. Control data,
       not transcript  -- NULL/__resume__ B-VALUE -> 'A-VALUE';
       task/__resume__ ['B-VALUE'] -> ['A-VALUE']
[PASS] it REPLACED rather than added: same conflict key, same row count  -- 3 rows before, 3 after
[PASS] both workers believe they succeeded; neither raised, neither logged
[PASS] no fork, no checkpoint added: this damage is NOT the dropped-write mechanism proof.py
       shows  -- 2 checkpoints at stage 0, 2 after B, 2 after A; forks=0
[PASS] the surviving run finished on a human input NEVER GIVEN TO IT: the frozen worker's
       'A-VALUE', not the live worker's 'B-VALUE'  -- log=['start', 'A-VALUE|SECOND']
```

Three things make this a distinct finding rather than a restatement of `README.md`:

1. **No fork, no added checkpoint.** The chain is byte-identical in shape before and after the
   zombie writes: 2 checkpoints throughout, 0 forked parents. The proven failure is *entirely* a
   fork story — a second child of the freeze-time tip, a losing branch, orphaned checkpoints.
   Here there is nothing to reconcile, nothing orphaned, and nothing a `parent_checkpoint_id`
   audit or an `aget_state()` reconciliation could notice. **The damage is invisible to every
   detector the proven failure suggests.**
2. **The value is not lost, it is substituted.** `DO NOTHING` deletes information; `DO UPDATE`
   replaces it with a plausible, well-typed, same-shaped *wrong* value. The row still
   deserialises. `get_state()` still reports a coherent pending interrupt.
3. **The consequence is not a malformed transcript, it is a wrong execution.** Stage 4 is
   identical code in both halves and it *completes normally*: `log = ['start', 'A-VALUE|SECOND']`.
   An operator approved `'B-VALUE'`, worker B told them so, and the thread executed `'A-VALUE'`.
   Nothing raised at any point, in any of the four processes involved.

The `__interrupt__` row is overwritten too, but invisibly: `Interrupt.from_ns` derives the id as
`xxh3_128_hexdigest(ns)` (`types.py:616-618`), so both workers produce a byte-identical
`Interrupt`. `__resume__` is where the divergence shows, because it carries the human's answer.

### Part 2b — the collision needs no instrumentation

Part 2's freeze point is reached with a cooperative gate (below). The obvious objection is that
the ordering was engineered. So Part 2b runs the same two resumes with **no gate, no `SIGSTOP`,
no fence** — two processes spawned within ~10 ms of each other, alternating which goes first — and
reports 5 trials:

```
trial 2  spawned A then B  rows at (NULL_TASK_ID,-4)=1  kept=['A-VALUE']
         then a later worker resumed with 'SECOND' -> ['start', 'A-VALUE|SECOND']
trial 3  spawned B then A  rows at (NULL_TASK_ID,-4)=1  kept=['B-VALUE']
         then a later worker resumed with 'SECOND' -> ['start', 'B-VALUE|SECOND']
which worker's value the thread kept, over 5 trials: {"['A-VALUE']": 3, "['B-VALUE']": 2}
```

The tally is **different on every run** of the script (observed 5–0, 4–1, 3–2 across runs) and does
not follow the spawn order. So: which human input the run finally executes on is settled by process
scheduling, and the losing worker is told nothing. Reported, not asserted — one thing here is not a
matter of timing and *is* asserted: `rows at (NULL_TASK_ID,-4)=1` in every trial. The two processes
always contend for **one row**; they never store one each. The gate in Part 2 only decides which of
them wins.

## Part 3 — the existing fence already blocks it

Verified, not assumed. [`fence.py`](fence.py)'s guard sits at `_cursor(pipeline=True)`, which is
*upstream of the statement choice* — `put_writes` picks its `ON CONFLICT` clause and only then
opens a cursor — so it should cover `DO UPDATE` exactly as it covers `DO NOTHING`. It does:

```
stage 3  worker A SIGCONT'd and allowed to write: ok=False
         error=DivisionByZero: division by zero sqlstate=22012
[PASS] worker B's claim is a compare-and-set: 1 -> 2 in one statement, and a later claim
       expecting 1 gets nothing  -- B claimed 2; stale re-claim -> None
[PASS] worker A's run terminated instead of writing
[PASS] the refusal came from POSTGRES, not a Python raise -- LOAD-BEARING
       -- sqlstate=22012
[PASS] the fence PREVENTS the overwrite: the survivor's control data is untouched, on the
       DO UPDATE statement as well as DO NOTHING
       -- NULL/__resume__='B-VALUE'  task/__resume__=['B-VALUE']
[PASS] the fence row advanced exactly once
[PASS] fenced, the run finishes on the input that was actually supplied to the live worker
       -- log=['start', 'B-VALUE|SECOND']
```

**This is a real strengthening of the fence's claim, not a footnote.** Before this run, the fence
was known to defend one statement against one kind of loss. It now demonstrably defends *both*
statements, and the second one against a failure mode with no observable signature. Nothing in
`fence.py` changed, and nothing needs to: choosing the `_cursor(pipeline=True)` seam — rather than
wrapping `put_writes` or filtering by channel — is what makes it clause-agnostic. That was a
Phase-0 decision made for a different reason (per-write rather than per-run commit granularity);
this is a second, independent payoff from it.

The corollary for the protocol SQL being drafted: **do not special-case the write.** A fence that
inspected the batch, the channel set, or the statement would have to enumerate `WRITES_IDX_MAP`
and would break the day upstream adds a fifth special channel. Guard the seam, not the statement.

## What this does not prove

- **The freeze point is reached with a cooperative gate, not by an unassisted `SIGSTOP`.** Worker
  A's saver stalls on a file at `_cursor(pipeline=True)` — the fence's own seam — so it loads
  checkpoint `C`, runs its node, and issues no thread-state SQL until released; the orchestrator
  then `SIGSTOP`s it and confirms `ps` state `T`. The stall is what makes the winner deterministic;
  the `SIGSTOP` is what makes it a real freeze. A plain `SIGSTOP` at an arbitrary instant would
  reach the same state only by luck. **Part 2b is the answer to this**: with no gate at all the
  collision still happens every time, and only the winner becomes a coin flip.
- **Only `__resume__` is shown end-to-end.** `__interrupt__` is overwritten in the same run but
  with an identical value, so it demonstrates nothing. `__error__` and `__scheduled__` are read in
  source and measured at the SQL level only.
- **Stages 0 and 4 run in the orchestrator process.** Only the two contending workers, A and B,
  are separate OS processes. Stage 4 is deliberately **unfenced in both halves** and byte-identical
  between them, so that the only difference between `['start','A-VALUE|SECOND']` and
  `['start','B-VALUE|SECOND']` is whether A's write was refused.
- **One fence branch.** Part 3 exercises the `supports_pipeline=True` path only.
  `README.md` covers the `conn.transaction()` fallback and documents that the connection-level
  `self.pipe` branch still refuses the write but surfaces `PipelineAborted`/`sqlstate=None`
  instead of `22012` — so a production fence must accept both error shapes. That divergence is
  untested here too.
- **Top-level graph, no subgraphs, no `deepagents`, no parallel tasks.** And the same environment
  caveats as `README.md`: one machine, one embedded Postgres, no pgbouncer, no partition, no clock
  skew.
- **`NULL_TASK_ID` is imported from `langgraph._internal._constants`** — a private module. Its
  value (`00000000-0000-0000-0000-000000000000`, `:93`) is a durable database key, which is a
  slightly awkward combination, but this file only reads it.
- **The final database state is not the asserted state.** Stage 4 mutates the thread after the
  last `check()` in each half, exactly as `probe_resume` does in `proof.py`. Inspecting
  `durable_agents_overwrite` after a run shows the post-stage-4 thread.

## Layout

| File | |
|---|---|
| `overwrite.py` | everything: the four parts, the worker (`--worker <json>`), the assertions, the verdict |
| `overwrite.out` | full stdout of a real run |
| `fence.py` | unchanged — reused as-is; that it needed no change is part of the result |

## Suggested edits to `README.md`

Not applied — `README.md` is out of this file's scope. Two places now say something weaker than
what is measured:

1. **"The other `ON CONFLICT` clause"** section, the closing paragraph
   ("The end-to-end version … is recorded as an untested exposure, not as a finding.") — this is
   now proven, and by a *simpler* vehicle than the parenthetical predicted.
2. **"What this does not prove"**, the bullet
   "**The `__interrupt__`/`__resume__` overwrite is an untested exposure** (above). The SQL
   mechanism is measured; the end-to-end scenario is not run." — no longer true.

Both can point here instead.
