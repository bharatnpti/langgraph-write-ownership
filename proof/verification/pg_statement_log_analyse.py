"""Parse the Postgres statement log slice: who attempted which write, and when.

Provenance: copied verbatim from the adversarial-review scratchpad as
`anlog.py`. Only the SLICE filename below was changed to match this
directory's copy of the log (renamed from `pg_slice.log` to
`pg_statement_log_slice.log`) -- see ../README.md for the evidence-to-claim
mapping. No other dependency: pure stdlib log parsing, no DB connection.
"""
from __future__ import annotations

import pathlib
import re
import sys

SLICE = pathlib.Path(__file__).resolve().parent / "pg_statement_log_slice.log"
HEAD = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \S+ \[(\d+)\] (\w+):  (.*)$")


def records():
    cur = None
    for line in SLICE.read_text(errors="replace").splitlines():
        m = HEAD.match(line)
        if m:
            if cur:
                yield cur
            cur = {"ts": m.group(1), "pid": m.group(2), "level": m.group(3),
                   "text": m.group(4)}
        elif cur:
            cur["text"] += "\n" + line
    if cur:
        yield cur


def main() -> None:
    thread = sys.argv[1] if len(sys.argv) > 1 else "half-a"
    recs = list(records())
    print(f"{len(recs)} log records; backends: "
          f"{sorted({r['pid'] for r in recs})}")

    # pair each statement LOG with the DETAIL that follows it on the same pid
    events = []
    for i, r in enumerate(recs):
        if r["level"] != "LOG":
            continue
        t = r["text"]
        detail = ""
        for j in range(i + 1, min(i + 3, len(recs))):
            if recs[j]["pid"] == r["pid"] and recs[j]["level"] == "DETAIL":
                detail = recs[j]["text"]
                break
        events.append((r["ts"], r["pid"], t, detail))

    def kind(t: str) -> str | None:
        for k in ("INSERT INTO checkpoint_writes", "INSERT INTO checkpoints",
                  "INSERT INTO checkpoint_blobs", "update proof_fence",
                  "select 1 / (select count(*)", "SELECT", "select"):
            if k in t:
                return k
        return None

    print(f"\n--- statements touching thread {thread!r} (writes + fence only) ---")
    seen_err = 0
    for ts, pid, t, detail in events:
        k = kind(t)
        if k is None or k in ("SELECT", "select"):
            continue
        if thread not in detail and "proof_fence" not in t:
            continue
        params = re.findall(r"\$(\d+) = ('[^']*'|NULL)", detail)
        pd = {int(a): b for a, b in params}
        if k == "INSERT INTO checkpoint_writes":
            print(f"  {ts} pid={pid}  put_writes  ckpt={pd.get(3, '?')[-10:]}"
                  f" task_id={pd.get(4, '?')[:10]} idx={pd.get(6)} channel={pd.get(7)}")
        elif k == "INSERT INTO checkpoints":
            print(f"  {ts} pid={pid}  put(ckpt)   ckpt={pd.get(3, '?')[-10:]}"
                  f" parent={pd.get(4, '?')[-10:]}")
        elif k == "INSERT INTO checkpoint_blobs":
            print(f"  {ts} pid={pid}  put(blob)   channel={pd.get(3)}"
                  f" version={pd.get(4, '?')[-12:]}")
        elif k == "update proof_fence":
            print(f"  {ts} pid={pid}  CLAIM       {detail}")
        elif k.startswith("select 1 /"):
            print(f"  {ts} pid={pid}  FENCE GUARD {detail}")

    print("\n--- ERROR/FATAL records ---")
    for r in recs:
        if r["level"] in ("ERROR", "FATAL", "WARNING"):
            seen_err += 1
            print(f"  {r['ts']} pid={r['pid']} {r['level']}: {r['text'][:160]}")
    if not seen_err:
        print("  (none)")


if __name__ == "__main__":
    main()
