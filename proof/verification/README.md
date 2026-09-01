# Verification evidence

**This directory is evidence, not code.** Nothing here is imported by `../proof.py`, by
`../fence.py`, or by any future library package. It exists so that three claims already
committed elsewhere in this repo stay checkable by someone who wasn't in the room:

| Committed claim | Where it's made |
|---|---|
| The checkpoint chain fork is general to LangGraph + `PostgresSaver`; the malformed message list is specific to `deepagents`' `DeltaChannel` | [`../README.md`](../README.md) "Two tiers of claim"; the pinned LangGraph fact base; `../../docs/open-issues/01-checkpointer-fencing.md`; the parent project's roadmap Phase 1, "Tiered claim: the fork is general to LangGraph; the malformed list is Deep Agents' DeltaChannel" |
| The proof survived an adversarial mutation pass — 8 of 9 mutations turned the suite red | the parent project's roadmap Phase 1, "Survived an adversarial pass — 8 of 9 mutations turned the suite red" |
| The `sqlstate == "22012"` assertion is load-bearing, and Half B passes for the right reason (a server-side refusal, not a client-side raise that happens not to be exercised by this scenario) | [`../README.md`](../README.md) "The `sqlstate` assertion is load-bearing, not corroboration" |

All three were originally checked by an adversarial review whose scripts lived only in a
scratchpad that gets cleared. This directory copies the scripts (fixed just enough to run from
here) and the raw outputs that can't be regenerated, and maps every file to the claim it backs.

Everything below was copied **verbatim** from that scratchpad unless a "Provenance" comment at
the top of the file says otherwise; every `.py` file here carries such a comment.

## The scripts — re-runnable evidence

Interpreter: `spike/.venv/bin/python` (3.12.13), from this directory, e.g.:

```bash
cd proof/verification
../../spike/.venv/bin/python tier_delta_vs_plain.py drive deep
```

All nine were actually run against this repo's pinned environment and embedded Postgres
(`spike/pgdata`) while writing this README. Every one exited 0.

| File | What it demonstrates (verified live, this session) | Backs |
|---|---|---|
| [`tier_delta_vs_plain.py`](tier_delta_vs_plain.py)<br>`drive deep` \| `drive plain` | The SIGSTOP/SIGCONT two-process takeover on a `--graph deep` vs `--graph plain` switch. Live: both fork (`fork_parent_is_freeze_tip=True` in both); `deep` reads back malformed (2 `orphans()` problems, **0** `messages` blobs); `plain` reads back well-formed (**0** problems, **9** `messages` blobs). | Tier 1 / Tier 2 split |
| [`tier_delta_vs_plain_first_pass.py`](tier_delta_vs_plain_first_pass.py) (no args) | The same comparison, first-pass version — both flavors in one process, full measurement table. Live: identical split (0 vs 9 blobs, 2 vs 0 orphans). | Tier 1 / Tier 2 split |
| [`fence_pipeline_branches.py`](fence_pipeline_branches.py) (no args) | The one `_cursor(pipeline=True)` branch (`self.pipe is not None`) that the two-process `proof.py` scenario never reaches, tested in isolation with a single `put_writes` call. Live: the SQL guard refuses (0 rows land, `sqlstate=22012`); a naive client-side raise after queueing commits anyway (1 row); an unfenced control always commits. | "Why the guard looks like that" — server-side-only refusal; see nuance below |
| [`write_tap.py`](write_tap.py) (no args) | Millisecond timeline of every `put`/`put_writes`/tool call, fenced and unfenced. Live, fenced: A's first post-thaw action is `PUT_WRITES`, which raises immediately (`DivisionByZero`/`22012`) — A never reaches `lookup(key='k2')`. Unfenced: A proceeds through `PUT_WRITES`, `PUT`, the tool call, and another `PUT_WRITES`/`PUT`. | "A write tap confirms `put_writes` is A's first post-thaw action" |
| [`queued_write_commits.py`](queued_write_commits.py) (no args) | A client-side raise after queueing, at each of the three `_cursor(pipeline=True)` shapes. Live: `self.pipe` set and `supports_pipeline=True` both **commit the queued write anyway**; only the `supports_pipeline=False` `conn.transaction()` fallback rolls it back. | Why the `sqlstate` assertion is load-bearing |
| [`misc_probes.py`](misc_probes.py) (no args) | E5: all three pipeline branches refuse a stale-fenced write. E7: `put_writes`'s two `ON CONFLICT` statements pick opposite winners on the same key (`DO NOTHING` -> first-writer-wins, `DO UPDATE` -> last-writer-wins), and which real channel combinations pick which. E8: `langchain_anthropic`, `ChatAnthropic._get_request_payload` and `convert_to_openai_messages` all accept the malformed list with no complaint. | "The other `ON CONFLICT` clause"; "nothing installed objects to the list" |
| [`independent_validators.py`](independent_validators.py) (no args, needs setup — see below) | Loads the real malformed thread out of the database and runs it through five validators outside the proof's own code. Live: `deepagents`' own `PatchToolCallsMiddleware` repairs the dangling `tool_use` and leaves the orphaned `tool_result` untouched; `langchain_anthropic`, `langchain_core`, and the `anthropic` SDK's `TypedDict` params all accept the list. (`langchain_google_genai`'s probe is skipped — an `ImportError` from a version mismatch in this environment, handled by the script's own `try`/`except`.) | "The independent witness: `deepagents`' own repair middleware" |
| [`pg_statement_log_analyse.py`](pg_statement_log_analyse.py) `[thread]` | Parses the raw Postgres statement log slice into a per-statement timeline. Live, against the preserved log: 500 records, 34 backend pids, exactly the `ERROR: division by zero` pairs the fence guard produces. Pure stdlib — no DB connection. | "Half B passes for the right reason" (statement-log side) |
| [`pg_statement_log_control.py`](pg_statement_log_control.py) `<on\|off> <db>` | The control utility that produced `pg_statement_log_slice.log`: turns Postgres statement logging on/off for one database and marks the byte offset to slice from. Code path smoke-tested this session — see caution below; not run in place. | Same evidence chain, as the tool rather than the evidence |

### Nuance: `fence_pipeline_branches.py` vs. the third `_cursor` branch caveat

[`../README.md`](../README.md) "What this does not prove" says the third `_cursor` branch
(`self.pipe` set), measured through the *full* two-process scenario, surfaces worker A's error as
`psycopg.errors.PipelineAborted` with `sqlstate=None` — not `DivisionByZero`/`22012`.
`fence_pipeline_branches.py` tests the same branch **in isolation** (one direct `put_writes` call)
and gets a clean `sqlstate=22012`. These are not in tension: `mutations/m7_conn_pipeline.txt`
(below) re-measures the same branch through the full scenario and reproduces the
`PipelineAborted`/`sqlstate=None` shape exactly — the surrounding pipelined operations in the full
scenario are what change the error shape the caller ultimately sees. Isolated and end-to-end
measurements of the same branch, agreeing on the refusal and disagreeing only on error shape, for
an explainable reason.

### Setup note: `independent_validators.py`

`main()` makes two `load()` calls. The first, `load("narrow-deep", "da_narrow_deep")`, is what the
rest of the script's probes run against — populate it by running `tier_delta_vs_plain.py drive
deep` first (done above; state persists in `spike/pgdata` after that). The second,
`load("half-b")` (db `da_tap`), was populated in the original review by a `proof.py` run pointed
at a `da_tap` database that isn't preserved standalone in this directory — but as verified while
writing this README, that state is **still there**: `spike/pgdata` is the same long-lived embedded
Postgres cluster the whole review ran against, and this task was explicit about not
reinitialising it. So today this second probe also reads back real data (an 8-message well-formed
thread), not an empty list. If `spike/pgdata` is ever reinitialised, only this one contrastive
`show()` print would go back to reading empty — none of the probes that matter for the claim
depend on it.

### Caution: `pg_statement_log_control.py`

This is a control utility, not a probe. `main()` runs `ALTER DATABASE ... SET log_statement =
'all'`, a **persistent, cluster-level config change** on whatever database you name, against the
shared embedded Postgres instance. It stays in effect until you run the script again with `off`
against the same database. It was verified this session only as `on <throwaway db>` immediately
followed by `off <same throwaway db>`, run from a temporary copy outside this directory so no
`logoffset.txt` marker was left behind here — the round trip confirmed clean (`pg_db_role_setting`
shows the two settings after `on`, and `None` after `off`). Point the copy in this directory at a
real scenario's database only if you deliberately want that database's statements logged.

## The two preserved full runs, and the statement log they bracket

| File | What it is | Backs |
|---|---|---|
| [`write_tap_proof_output.out`](write_tap_proof_output.out) | stdout of a full `proof.py` run against a dedicated `da_tap` database, from the write-tap investigation session (same `HALF A` / `HALF B` structure as [`../proof.out`](../proof.out)). | "Half B passes for the right reason" (write-tap side) — supporting, not the tap itself (`write_tap.py`'s live run is) |
| [`pg_statement_log_proof_output.out`](pg_statement_log_proof_output.out) | stdout of a full `proof.py` run against a dedicated `da_log` database — the run whose SQL Postgres's own log captured. Timestamp-correlated with the slice below (both 2026-08-30, 01:48-01:52). | "Half B passes for the right reason" (statement-log side) |
| [`pg_statement_log_slice.log`](pg_statement_log_slice.log) | The raw Postgres server log, sliced from the recorded byte offset through the run above. 1,555 lines, 34 backend pids, 6 `ERROR`/`FATAL`/`WARNING` records (two matched pairs of `division by zero`, from the guard firing at `half-a`/`half-b`/`half-b-nopipe`). | "Half B passes for the right reason" (statement-log side) — the primary artifact; `pg_statement_log_analyse.py` is how to read it |

Both `.out` files are **output-only**: the exact `proof.py` invocations that produced them (a
`da_tap`- and `da_log`-pointed variant of `proof.py`, not parameterized by database name) are not
preserved standalone. Regenerating the log slice means: `pg_statement_log_control.py on <db>`,
running a `proof.py`-shaped scenario against that database, `pg_statement_log_control.py off <db>`,
then slicing `spike/pgdata/log` from the recorded offset.

## `mutations/` — the adversarial pass

Mutation *source* is deliberately not preserved (see "Not copied" in the task that produced this
directory) — `mkmutants.py` built each mutant from **this repo's actual, currently-committed**
`proof/fence.py` / `proof/proof.py` / `proof/worker.py` (confirmed: its `SRC` is this repo's
`proof/` directory, hardcoded), so what follows is a faithful adversarial pass against the current
fence, described precisely instead of shipped as code:

| File | Mutation | Result |
|---|---|---|
| [`mutations/m1_guard_always_pass.txt`](mutations/m1_guard_always_pass.txt) | `GUARD_SQL` replaced so the fence guard always passes | CAUGHT — 12 assertions failed |
| [`mutations/m2_claim_no_bump.txt`](mutations/m2_claim_no_bump.txt) | `CLAIM_SQL` changed to `SET fence = fence` (the claim no longer bumps the token) | CAUGHT — 16 assertions failed |
| [`mutations/m3_trivial_fence_compare.txt`](mutations/m3_trivial_fence_compare.txt) | guard `WHERE` made trivially true with `or true` | CAUGHT — 12 assertions failed |
| [`mutations/m4_no_unwind.txt`](mutations/m4_no_unwind.txt) | the `outer.__exit__` unwind removed before re-raise | CAUGHT — but only by the `supports_pipeline=False` half, which **hung as designed** (confirmed: the file ends mid-run, killed after 120s with `FileNotFoundError` on a result file that was never written, not a clean `VERDICT`); the pipeline half passed 8/8 |
| [`mutations/m5_python_raise.txt`](mutations/m5_python_raise.txt) | fence read through the same pipelined cursor, with a Python raise | CAUGHT incidentally — confirmed: this variant aborts before worker A even reaches its 2nd model call ("worker A never reached its 2nd model call; aborting"), so it never becomes a valid test of the constraint |
| [`mutations/m6_raise_after_queue.txt`](mutations/m6_raise_after_queue.txt) | write queued, then a **client-side** Python raise, fence read on a separate connection | **MISSED** by everything except the `sqlstate == "22012"` assertion — confirmed: `VERDICT: FAILED -- 2 assertion(s)`, and both are that one assertion (once per Half-B variant) |
| [`mutations/mA_no_worker_b.txt`](mutations/mA_no_worker_b.txt) | worker B never spawned | CAUGHT — 15 assertions failed, confirmed no fork |
| [`mutations/mA_b_other_thread.txt`](mutations/mA_b_other_thread.txt) | worker B pointed at a different `thread_id` | CAUGHT — confirmed no fork, though the harness itself then hits an unhandled `KeyError` while formatting its own report (`b_out` has no `"messages"` key, since B "succeeded" on the wrong thread) — after the relevant `[FAIL]` lines had already printed |
| [`mutations/mA_no_freeze.txt`](mutations/mA_no_freeze.txt) | `SIGSTOP` removed so A completes before B resumes | CAUGHT — 11 assertions failed, confirmed 0 forked parents |

That is the nine. Two more files exist in this folder because the copy instruction was "all
`mutout/*.txt`," and honesty means not hiding them:

- [`mutations/m7_conn_pipeline.txt`](mutations/m7_conn_pipeline.txt) — a **tenth** mutation, not
  one of the canonical nine above. Diffing the mutant against its pre-mutation base shows it
  repurposes the worker's `force_no_pipeline` flag to instead wrap the saver's connection in a
  connection-level `conn.pipeline()` — i.e. it exercises the third `_cursor` branch end-to-end
  instead of in isolation. Result: `VERDICT: FAILED -- 1 assertion(s)` — "the refusal came from
  Postgres, not from a Python raise (corroboration only) -- sqlstate=None." See the nuance note
  above: this is independent, complementary evidence for the `PipelineAborted` caveat in
  `../README.md`, not a contradiction of `fence_pipeline_branches.py`'s clean `22012`.
- [`mutations/exits.txt`](mutations/exits.txt) — a **partial, earlier** manifest: 8 entries
  (`EXIT=1` each, matching the nine above minus `m6`), timestamped before `m6` and `m7` were run.
  It predates the full set of ten `.txt` outputs actually in this folder.

Not copied, and not in this manifest: `mutants/m8_a_hangs`, in the scratchpad, whose output was
never filed into `mutout/` and so falls outside even the "all `mutout/*.txt`" instruction — it
mutates `worker.py` so A hangs forever after the thaw (a deadlock stand-in), which tests the
*harness's* hang-handling, not the fence's correctness. Noted here only for completeness.

## `determinism/`

Two independent repeat-run checks, from two different points in the review:

- [`determinism/run01.txt.norm`](determinism/run01.txt.norm) through
  [`determinism/run08.norm`](determinism/run08.norm) (9 files) — repeats against an **earlier**
  revision of `proof.py` (pre-Tier-split: `VERDICT: PROVEN` with 23 `[PASS]`, 0 `[FAIL]`, no "in
  two tiers" language — the wording matches the `pristine/proof.py` snapshot the other scripts in
  this directory reference, not the current `../proof.py`). Confirmed: after normalizing away
  timestamps, pids, checkpoint ids and task ids, all 9 are byte-identical **except** one line the
  normalizer's own scrubbing evidently misses — a Python object's memory address inside a stderr
  message (`<psycopg.Pipeline ... at 0x10fe2dbb0>`). Stripping that one line too makes all 9
  hash identically. The normalizer script itself was not preserved to the scratchpad; its effect
  is directly verifiable by diffing any `.norm` file against its raw counterpart.
- [`determinism/exits.txt`](determinism/exits.txt) — partial manifest, 5 of the raw runs
  (`run02`-`run06`, all `EXIT=0`).
- [`determinism/det1.out`](determinism/det1.out),
  [`determinism/det2.out`](determinism/det2.out),
  [`determinism/det3.out`](determinism/det3.out) — three **full, unnormalized** reruns against
  the **current, committed** `../proof.py` (confirmed: each contains the literal
  `VERDICT: PROVEN, in two tiers` line and exactly 25 `[PASS]` / 0 `[FAIL]`, matching
  `../proof.out`'s own claimed count).

Together: 12 preserved repeat runs (9 normalized + 3 raw), spanning a revision boundary, every one
landing on the identical pass/fail pattern and verdict. The proof's *outcome* is deterministic
even though pids, timings and generated ids are not.

**Worth flagging directly: [`../proof.out`](../proof.out) — the file `../README.md` cites as
"full stdout of a real run, 25 assertions, 0 failures" — is currently 0 bytes in this working
tree.** That is outside this directory's scope to fix (this task only touches
`proof/verification/`), but `determinism/det1.out`, `det2.out` and `det3.out` above independently
substantiate the exact claim `../proof.out` is supposed to and currently doesn't: each is a full
run of the same current `proof.py`, each shows 25/0/`PROVEN, in two tiers`. Worth running
`spike/.venv/bin/python proof/proof.py` again to regenerate `../proof.out` itself.

## Not copied: the conformance suite

`conf/` and `confpkg/` in the scratchpad are vendored copies of upstream LangGraph's
`langgraph-checkpoint-conformance` test suite — third-party source, not reproduced here (licensing
and noise). The conformance evidence they produced is already committed at
[`../../spike/q_conformance.out`](../../spike/q_conformance.out) and
[`../../spike/q_conformance_cap.out`](../../spike/q_conformance_cap.out). Reproducing it means
re-fetching the suite from `langchain-ai/langgraph` with `gh` and re-running it against the pinned
versions in [`../README.md`](../README.md) "Pinned" — not something to vendor into this repo.

Also not copied, per the task's instructions: `_loop.py` / `base_init.py` (installed-package
source, for reading only), and `mut/` / `tapped/*.py` / `logged/*.py` (further mutated or
instrumented copies of the proof — the outputs in this directory are the evidence; the Python
isn't reproduced).

## Provenance and safety

- Every `.py` file in this directory carries a "Provenance" note at the top saying whether it was
  copied byte-for-byte or had its `sys.path` / self-re-exec filename fixed, and why. Four scripts
  needed a path fix (`tier_delta_vs_plain.py`, `fence_pipeline_branches.py`,
  `independent_validators.py`, `pg_statement_log_analyse.py`) because the originals pointed at the
  scratchpad's throwaway `pristine/` copy of `fence.py` or at their own pre-rename filename; the
  other five already hardcoded this repo's absolute path and needed no change.
- Every file copied into this directory — every `.py`, every `.out`, every `.log`, every `.txt` —
  was scanned for API keys, tokens, passwords and connection-string credentials before being
  written here. **None were found.** The only `api_key=` occurrences in this directory
  (`misc_probes.py`, `independent_validators.py`) are obviously-fake placeholder strings
  (`"sk-not-a-real-key"`, `"not-a-real-key"`) that `ChatAnthropic` requires be non-empty; they are
  not, and were never, real credentials.
- The scratchpad root also contains a file named `llm_key` (mode 600). It is not named in the copy
  instructions for this directory and was **not** opened or copied.
