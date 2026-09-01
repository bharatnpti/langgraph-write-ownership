# The zombie proof

Originally Phase 1 of a separate crash-recovery library; kept here as standalone evidence.
Two OS processes,
one `thread_id`, a real Deep Agents loop. Throwaway code, kept forever as evidence: this is the
reproduction an upstream LangGraph maintainer will ask for, and the regression test the library
gets built on.

## Run it

```bash
spike/.venv/bin/python proof/proof.py     # ~15s, exit 0 on PROVEN
```

One command. **Do not redirect stdout into `proof.out`** — the script opens that file
itself, so `proof.py > proof.out` puts two writers on one path and truncates it.

No API keys, no Docker, nothing to install: the model is a scripted
`BaseChatModel` and Postgres is the embedded `pgserver` cluster in `spike/pgdata/`. Idempotent —
it drops and recreates its own database objects (`durable_agents_proof`) on every run.

Full stdout of a real run: [`proof.out`](proof.out). 25 assertions, 0 failures. Any `FAIL` line
makes the process exit 1.

## Two tiers of claim — read this before quoting anything below

The two halves of the damage do **not** have the same blast radius, and conflating them would
make the whole report easy to dismiss. So they are separated everywhere: in the assertion
headers, in the verdict, and here.

> **Tier 1 — general to LangGraph + `PostgresSaver`. Nothing to do with `deepagents`.**
> Two workers running the same `thread_id` fork the checkpoint chain at the checkpoint that was
> the tip when the first worker stalled. One worker's `put_writes` is **silently discarded** while
> that worker reports success. The loser's committed work — its terminal checkpoint included —
> becomes **unreachable from the tip**. The answer it returned to its caller **contradicts** the
> thread. Both workers exit 0. Nothing raises, nothing logs.

> **Tier 2 — specific to `deepagents`' `DeltaChannel` transcript.**
> On top of Tier 1, the surviving branch's message list is **malformed**: it holds a `tool_result`
> whose `tool_use` was discarded and is stored *nowhere*, next to a `tool_use` whose result is on
> the abandoned branch. This is a consequence of the *delta* representation, which makes the
> dropped write **unrecoverable** rather than merely superseded.

A plain `StateGraph` on `MessagesState` (`Annotated[list, add_messages]`, a
`BinaryOperatorAggregate`) **forks in exactly the same way and leaves each branch internally
consistent**. Both scenarios were run, two OS processes each, identical freeze/thaw:

| measured | `deepagents` (`DeltaChannel`) | plain `MessagesState` |
|---|---|---|
| chain forks, parent = the freeze-time tip | yes | **yes** |
| one `messages` row at the fork parent, same `task_id` | yes | **yes** |
| loser's checkpoints unreachable from the tip | 3 | **3** |
| loser's answer returned to its caller but not on the tip | yes | **yes** |
| `checkpoint_blobs` rows for `messages` | **0** | **9** |
| read-back list malformed | 2 problems | **none — well-formed** |

The `0` vs `9` is the whole reason for the difference. A `DeltaChannel` writes **no** full-list
blob, so the durable transcript *is* the `checkpoint_writes` rows — drop one and the message it
carried exists nowhere. A `BinaryOperatorAggregate` writes the whole accumulated list as a blob at
every step, so the losing worker's list survives intact in *its own* blob; its branch wins the tip
and reads back as a consistent conversation. What is lost there is the *other* worker's work,
silently — Tier 1 — not the list's integrity.

So: **the fork is LangGraph's. The malformed list is `deepagents`'.** Both matter; only the first
generalises.

## Pinned

A verdict here says nothing about any other version.

| | |
|---|---|
| Python | 3.12.13 |
| `langgraph` | 1.2.11 |
| `langgraph-checkpoint` | 4.2.0 |
| `langgraph-checkpoint-postgres` | 3.1.2 |
| `deepagents` | 0.7.11 |
| `langchain-core` | 1.6.1 |
| `psycopg` | 3.3.4 (libpq 180000, so `has_pipeline()` is True) |
| PostgreSQL | 16.2, embedded via `pgserver` 0.1.4 |

## The scenario

Shared by both halves, so the only variable is the fence.

1. **Worker A** starts the run: `invoke({"messages": [...]}, config, durability="sync")`.
   It gets through one full tool round-trip.
2. Inside its **second model call** A writes a pid file and blocks. The orchestrator
   **`SIGSTOP`s** it and confirms `ps` reports state `T`. Frozen, not killed — a frozen worker is
   the failure being defended against, and a kill cannot demonstrate a thaw.
   At this moment checkpoint `C` (step 3) is durable and carries no writes yet.
3. **Worker B** — a separate OS process — takes the thread over with `invoke(None, config)`.
   It writes nothing to the thread first: no `update_state`, no fork. It re-runs the same model
   task, gets a *different* continuation (a real LLM would; scripting it is the faithful
   stand-in), and finishes the run.
4. A is **`SIGCONT`'d** and allowed to write.

### Tier 1: why the chain forks and a write vanishes

Under `durability="sync"`, the checkpoint for super-step *k* is committed before step *k*'s tasks
run, and each task's output lands as `put_writes(C_k, task_id)` where `task_id` is deterministic —
derived from the checkpoint namespace, the node and the task path, not from the process.

That gives the collision:

- Both A and B run the **same model task at the same checkpoint `C`**, so both call
  `put_writes(C, task_id)` with the *same* `task_id`. Measured directly with a write tap: two
  separate OS processes issue `put_writes` for the identical `(checkpoint_id, task_id)` pair.
- `put_writes` chooses between two statements, and for `messages` it picks
  `INSERT_CHECKPOINT_WRITES_SQL` — `INSERT … ON CONFLICT (thread_id, checkpoint_ns,
  checkpoint_id, task_id, idx) DO NOTHING`. That is **first-writer-wins**. B commits first;
  **A's `AIMessage` is silently discarded**, with no error and no warning. Observed: one stored
  row at `C`, carrying `tool_calls=['tc_B2']`. (The other statement, and the second hazard it
  hides, is [below](#the-other-on-conflict-clause).)
- A's *in-memory* channel still holds its own `AIMessage(tool_calls=['tc_A2'])`. A proceeds to
  write its own child of `C` — **the chain forks** — and to execute the tool call `tc_A2`,
  anchoring `ToolMessage(tool_call_id='tc_A2')` on its own branch.
- Checkpoint ids are time-ordered, so A's branch (written last) becomes the tip and B's
  terminal checkpoint becomes unreachable. Asserted, not just observed: B's terminal checkpoint
  is named and checked for membership in the orphaned set.

Two further consequences worth naming in a bug report:

- **Both workers returned success.** Neither raised, neither logged. B told its caller
  `'final answer from B'`; the thread's tip says `'final answer from A'`.
- **Nothing in the schema objects.** The complete constraint inventory of the checkpoint tables
  is four primary keys — no unique constraint on `parent_checkpoint_id`, no version column, no
  trigger, no FK.

### Tier 2: why that leaves `deepagents`' list malformed

`deepagents` declares `messages` as a **`DeltaChannel`**
(`deepagents/graph.py:73`, `snapshot_frequency=50`). A `DeltaChannel` stores no full-list blob;
the message list is reconstructed by replaying the `checkpoint_writes` rows along the parent
chain. So the durable representation of the transcript *is* those write rows — measured: **zero**
`checkpoint_blobs` rows for `messages` in the whole thread.

Replaying the tip therefore concatenates **B's `AIMessage` at `C`** with **A's `ToolMessage` at
A's child**:

```
HumanMessage   go
AIMessage      looking up k1     tool_calls=['tc_A1']
ToolMessage    value-of-k1                            tool_call_id=tc_A1
AIMessage      worker B turn 2   tool_calls=['tc_B2']      <-- B's branch
ToolMessage    value-of-k2                            tool_call_id=tc_A2   <-- A's branch
AIMessage      final answer from A
```

Every `tool_use` id in the thread was traced through the database — every `checkpoint_writes` row
*and* every `checkpoint_blobs` row for `messages`. The result is asserted, not narrated:

- `tc_A2` is offered by **no stored message anywhere in the thread**. The `AIMessage` that
  offered it is not superseded, not on a losing branch, not recoverable by rewinding — it was
  **annihilated** by `ON CONFLICT DO NOTHING`.
- Yet `ToolMessage(tool_call_id='tc_A2')` *is* durable, on A's branch, on the tip's chain. A
  `tool_result` can only be there if A ran the same task at the same checkpoint and its
  `put_writes` was dropped — which is how the proof observes both attempts from state alone.
- `tc_B2` is the mirror image: a **`tool_use` on the tip whose result is on the abandoned
  branch**, never answered.

This is durable state, not a read artifact.

### The independent witness: `deepagents`' own repair middleware

The strongest corroboration is inside `deepagents` itself. `PatchToolCallsMiddleware`
(`deepagents/middleware/patch_tool_calls.py`) is in the compiled graph — `create_deep_agent`
installs it in every branch of its graph assembly (`deepagents/graph.py:673`, `:758`, `:843`).
Run against the forked list, it **fires**:
it sees a dangling `tool_use`, rewrites the entire list with
`RemoveMessage(REMOVE_ALL_MESSAGES)`, and injects a synthetic

```
ToolMessage(tool_call_id='tc_B2',
            content="Tool call lookup with id tc_B2 was cancelled - another message came in
                     before it could be completed.")
```

Three things follow, all measured and printed by the proof (Assertion 4 in Half A):

1. **The framework itself classifies this thread as damaged.** Not our validator — theirs. And
   its own diagnosis of the cause is accurate: another message came in first.
2. **It repairs only half.** The dangling `tool_use` gets an answer; the orphaned `tool_result`
   (`tc_A2`) has no `tool_use` to synthesise and **survives the repair untouched**.
3. **The repair is itself committed to the thread**, as a `checkpoint_writes` row on the tip's
   chain. The thread is now durably poisoned in a shape the framework has already tried, and
   failed, to fix.

### Is the thread unresumable? — stated plainly, not overclaimed

**No, not mechanically.** LangGraph reads, replays and resumes the forked thread without
complaint, and `invoke(None, config)` returns normally. `.next` is empty: as far as LangGraph is
concerned the run finished.

What is destroyed is the thread's **usability at the next model call**. The proof shows a concrete
failure by driving the *next user turn* through a model that enforces the provider contract
(every `tool_result` needs its `tool_use`, and vice versa):

```
next user turn -> RuntimeError: provider would reject this request:
                  tool_result tool_call_id='tc_A2' has no matching tool_use ...
```

**That validator is ours** (`fence.orphans`), not a provider's, and it is not asserted — it is
reported. Be precise about what is and is not established here:

- **Nothing installed objects to the list.** `langchain_anthropic._format_messages`,
  `ChatAnthropic._get_request_payload` and `langchain_core`'s `convert_to_openai_messages` all
  accept it without complaint, and the `anthropic` SDK's request params are `TypedDict`s, which
  do no runtime validation. There is no client-side gate to trip.
- **The defensible claim is about the emitted payload shape.** `_get_request_payload` on the
  forked list emits, in order, `tool_use tc_A1`, `tool_result tc_A1`, `tool_use tc_B2`,
  `tool_result tc_A2` — i.e. a `tool_result` whose `tool_use_id` appears as **no `tool_use` in
  the request**, and a `tool_use` with no result. That is the shape both the Anthropic and the
  OpenAI APIs document as invalid.
- **✅ MEASURED against a real provider (2026-08-30).** This was the one claim here argued from a
  documented contract rather than observed. It is now observed. `provider_check.py` reads both
  threads out of the database this proof leaves behind — not retyped — converts them verbatim to
  chat-completions format with no repair or reordering, and posts them:

  | thread | tool_result offered by no preceding tool_use | HTTP |
  |---|---|---|
  | `half-a`, fencing **off** | `tc_A2` | **400** |
  | `half-b`, fencing **on** (control) | none | **200** |

  The provider's own words, on `half-a`:

  > Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.

  with `param = messages.[5].role` — and index 5 is exactly the orphaned `tool_result tc_A2` this
  file identifies. The provider independently locates the same message.

  The control is what makes it evidence: the fenced thread, sent through the identical code path,
  returns 200 with a real completion. So the 400 is attributable to the corruption, not to how the
  request was built.

  Re-run it yourself (no key is stored in this repo; the script skips cleanly without one):

  ```bash
  PROVIDER_KEY_FILE=/path/to/a/mode-600/key spike/.venv/bin/python proof/provider_check.py
  ```

  Measured on `gpt-4o-mini` via `https://eu.api.openai.com/v1/chat/completions`. Full output:
  [`provider_check.out`](provider_check.out).

## Half B — fencing on

Same script, same scenario, `--fence` on, run **twice**: once on the pipeline path and once with
`saver.supports_pipeline` forced `False`.

Asserted, nine times per variant: B's claim is a compare-and-set (fence `1 → 2` in one statement,
and a later claim expecting `1` gets nothing); **zero** checkpoint rows were added after the
takeover; the chain did not fork; worker A's run terminated instead of committing; **worker A's
failure carried SQLSTATE `22012`**; the fence advanced exactly once; B's message list is
well-formed *and* has the expected six-message shape; B's answer is the tip; and the thread
**accepts a next turn**.

### The `sqlstate` assertion is load-bearing, not corroboration

An earlier version of this file called the `sqlstate == "22012"` check corroboration only. That
was wrong, and a mutation test proved it. Mutate the guard so that it **queues the write and then
raises client-side in Python**, reading the fence on a separate connection — a fence that violates
this file's own primary constraint — and **7 of the 8 other Half B assertions still pass**. The
chain does not fork, no checkpoint rows are added, the fence advances once, B's thread is intact.
The `sqlstate` check is the *only* thing that goes red.

Why the state assertions miss it: the client-side raise stops worker A inside `put_writes`, whose
`INSERT … DO NOTHING` collides with the row B already wrote, so nothing new appears; and A never
reaches the `put` that would have written the forked checkpoint. The mutant looks safe *in this
scenario* while being unsafe in general. Measured separately, on a fresh key so nothing masks the
row: a client-side raise inside an open pipeline block **still commits the already-queued write**
on both pipeline branches — `psycopg`'s `Pipeline.__exit__` enqueues a `Sync` in `_exit_gen`
regardless of the exception. (Only the `supports_pipeline=False` `conn.transaction()` fallback
rolls it back.) A client-side fence that checked the fence one statement later would have
committed the fork.

So the error *shape* is the only signal that distinguishes a server-side fence from a client-side
one, and it is asserted.

### Why the guard looks like that

Three files, ~600 lines, and almost every odd-looking line is load-bearing. Each of these came out
of Phase 0 (the pinned LangGraph fact base) and is
honoured here:

| Constraint | How this proof honours it |
|---|---|
| The fence must be enforced in SQL, not by raising — a saver-raised rejection is silently dropped in 28 of 56 configurations | the guard is a SQL statement; every assertion is on database state |
| It must fail **server-side**: a Python `raise` inside a pipeline block still commits the queued writes (psycopg enqueues a `Sync` in a `finally`) | `GUARD_SQL` errors in Postgres; the `raise` is a consequence, not the mechanism — and the `sqlstate` assertion is what enforces it |
| A literal `1/0` is constant-folded | the divisor is a subquery. Re-measured every run in the pre-flight — see the correction below |
| Never read the verdict from bare `cur.rowcount` — it reads `-1` inside a pipeline | the guard reads no verdict at all; `claim()` uses `UPDATE … RETURNING` + `fetchall()` |
| A naive synchronous raise in the guard **deadlocks** — `PostgresSaver.lock` is never released | `_Guarded.__enter__` unwinds `outer.__exit__` before re-raising, and the `supports_pipeline=False` half is the regression test for it |
| The claim must be one statement whose `WHERE` tests exactly the column it writes | `CLAIM_SQL`: `SET fence = fence + 1 … WHERE fence = %s`. That shape is a property of the source, which no runtime predicate can test; what the proof asserts is the observable consequence — compare-and-set semantics |
| The fence row must live in the same database as the checkpoint tables | `proof_fence` is created next to them by `setup_database()` |
| `durability="sync"`, never `"exit"` | passed on every `invoke` |
| The resume path must write **nothing** before `invoke(None, config)` | worker B claims in *our* table, then invokes; no `update_state`, no fork |
| `setup()` needs a separate autocommit connection, once (`CREATE INDEX CONCURRENTLY`) | `setup_database()` |

**Per-write, not per-run.** Phase 0 also found that wrapping a whole run in one transaction fences
perfectly and is *wrong*: checkpoints only become visible at commit, so a worker that dies mid-run
leaves its successor nothing to resume from. This proof fences at
`PostgresSaver._cursor(pipeline=True)` — the seam passed `pipeline=True` at exactly three sites:
`put` (`__init__.py:321`), `put_writes` (`:368`), `delete_thread` (`:390`) — so each super-step
commits incrementally and a superseded worker is refused per write. A visible side benefit in the
transcript: worker A holds no lock while frozen, so worker B never blocks behind it.

The seam is **every write of *thread state***, which is not the same as every write. Of the four
remaining `_cursor()` calls, three are reads (`list` `:159`, `get_tuple` `:237`,
`get_delta_channel_history` `:490`/`:545`) — but `setup()` (`:92`) uses an unguarded `_cursor()`
and *writes*. Measured by tapping the cursor on a fresh database: `setup()` issues **22 statements
through `_cursor(pipeline=False)`, 20 of them writes** — the DDL migrations (four `CREATE TABLE`,
two `ALTER TABLE`, three `CREATE INDEX CONCURRENTLY`) plus ten
`INSERT INTO checkpoint_migrations (v) VALUES (%s)`. Harmless in substance — schema, not thread
state, and it runs once before any worker starts — but a claim that "every other `_cursor()` call
in that module is a read" is simply false, and this file used to make it.

### The other `ON CONFLICT` clause

`put_writes` does **not** always use `DO NOTHING`. It picks:

```python
query = (self.UPSERT_CHECKPOINT_WRITES_SQL          # ON CONFLICT ... DO UPDATE
         if all(w[0] in WRITES_IDX_MAP for w in writes)
         else self.INSERT_CHECKPOINT_WRITES_SQL)    # ON CONFLICT ... DO NOTHING
```

with `WRITES_IDX_MAP = {'__error__': -1, '__scheduled__': -2, '__interrupt__': -3,
'__resume__': -4}` (`langgraph/checkpoint/postgres/__init__.py:363-367`,
`langgraph/checkpoint/base/__init__.py:795`).
So the first-writer-wins behaviour above holds for `messages` and for every mixed batch, but a
batch consisting **only** of those four control channels is **last-writer-wins**.

That is a second, distinct hazard: a zombie's `__interrupt__` or `__resume__` write does not get
dropped — it **overwrites** the survivor's. The pre-flight measures the mechanism directly, running
both real statements twice against one conflict key:

```
INSERT .. DO NOTHING   -> survivor is 'survivor'
UPSERT .. DO UPDATE    -> survivor is 'zombie'
```

The end-to-end version — two workers both interrupted at the same checkpoint and task, the
zombie's resume value clobbering the survivor's — **is now demonstrated**, in
[`OVERWRITE.md`](OVERWRITE.md) and [`overwrite.py`](overwrite.py). It needed neither a tool nor
`deepagents`: a plain `StateGraph` with two `interrupt()` calls is enough, so the guess recorded here
originally (that it would need an interrupt inside a tool) was wrong in the reassuring direction.

It is a **different mechanism** from the dropped write above, and in one respect a worse one: **no
fork and no added checkpoint**, so the chain shape is identical before and after and nothing a
`parent_checkpoint_id` audit or an `aget_state()` reconcile could notice. The value is not deleted, it
is substituted with a well-typed plausible wrong one, and the surviving run then finishes on a human
input that was never given to it. In a normal human-in-the-loop cycle **4 of 6** `put_writes` batches
take the `DO UPDATE` path, so this is ordinary traffic rather than an API corner.

### A correction to a Phase 0 constraint

Phase 0 recorded that "a literal `1/0` in a `CASE` arm is constant-folded by Postgres and raises
unconditionally". Measured here on 16.2, that is **true only when the `CASE` condition is not
itself constant** — the pre-flight prints both:

```
CASE WHEN <subquery>=1 THEN 1 ELSE 1/0 END, fence MATCHES  -> DivisionByZero sqlstate=22012
CASE WHEN true THEN 1 ELSE 1/0 END (constant condition)    -> 1
```

With a real (non-constant) condition the arm is folded and raises even at `EXPLAIN` time, exactly
as warned — so the guidance is right for any guard anyone would actually write. With a constant
condition the planner drops the whole `CASE` before simplifying the arm, so the naive test of the
claim comes back green and misleads. Worth the one-line refinement in the reference.

## Layout

| File | |
|---|---|
| `proof.py` | orchestrator: spawns the workers, sends the signals, snapshots the chain, asserts, prints the verdict |
| `worker.py` | one worker process: a `deepagents` react loop against one `thread_id` |
| `fence.py` | the fence — one table, one claim statement, one guard, plus the message-list validator |
| `proof.out` | full stdout of a real run |

Deliberately absent, per the scope discipline in `strategy.md`: no runner, no `RunStore` protocol,
no recovery scanner, no state machine, no event log. One table.

## What this does not prove

- ~~No real provider 400.~~ **Closed** — measured as HTTP 400 with a 200 control, see above.
  What remains unmeasured is only whether *every* provider rejects it; one does, definitively.
- **Fencing protects state, not the outside world.** A fence cannot undo a side effect that has
  already happened — nothing can, and that is the permanent residue of at-least-once
  (ADR-0007). But **this run does not exhibit
  it**, and an earlier version of this file wrongly said it did. Worker A is frozen *inside* the
  model call, so after the thaw it is refused at its very first write: a write tap confirms
  `put_writes` is A's first post-thaw action, the guard rejects it, and A **never executes
  `tc_A2`'s tool at all** (`lookup` is called for `k1` only). The one tool call A did make ran
  before the freeze and was legitimate. A scenario that freezes *inside a tool* would exhibit the
  duplicate side effect; this one does not.
- **The third `_cursor` branch is untested, and would fail this proof's error-shape assertion.**
  `_cursor(pipeline=True)` has three branches: `self.pipe` set (a connection-level `Pipeline`, as
  `from_conn_string(..., pipeline=True)` produces), `supports_pipeline=True`, and the
  `conn.transaction()` fallback. The proof exercises the last two. Measured on the first: the
  fence **still refuses the write** — no fork, no rows added, fence advances once, the survivor's
  thread intact — but worker A's error surfaces as `psycopg.errors.PipelineAborted` with
  `sqlstate=None` instead of `DivisionByZero`/`22012`. So a *working* fence fails the `sqlstate`
  assertion there. A production fence needs to accept both shapes; this proof does not, on purpose,
  because loosening the check is exactly what the mutation test says not to do.
- ~~The `__interrupt__`/`__resume__` overwrite is an untested exposure.~~ **Closed** — proven end to
  end in [`OVERWRITE.md`](OVERWRITE.md), 19 assertions. Only `__resume__` was driven end to end;
  `__error__` and `__scheduled__` remain source-read and SQL-measured only.
- **The final database state is not the asserted state.** In each half, `probe_resume` runs after
  the last `check()` and *mutates the thread*: it commits a `HumanMessage`, `deepagents`' repair
  middleware runs and commits its rewrite, and a rejected turn leaves an `__error__` write and
  `.next == ('model',)`. That is deliberate — the post-probe state is the evidence for the
  middleware witness above — but if you inspect `durable_agents_proof` after a run, you are
  looking at the thread *after* the probes, not at what the assertions saw.
- **One machine, one embedded Postgres, no pgbouncer**, no network partition, no clock skew.
  The two workers are genuinely separate OS processes and the freeze is a genuine `SIGSTOP`, but
  transaction pooling would very likely change the picture and is untested.
- **Top-level only.** Phase 0 measured that recovery inside a Deep Agents *subagent* loses the
  whole in-flight super-step, including siblings already committed. This scenario does not use
  subagents.
- **One fork shape.** A frozen model call is the sharpest case because the two workers collide on
  one `task_id`. Freezing inside a tool, or with parallel tool calls, will fork the chain too but
  may leave each branch internally consistent.
