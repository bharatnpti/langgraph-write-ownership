"""E3 -- verify CORRECTION 2.3 and 4.4 by tapping every write and tool call.

Runs the Half-B (fenced) scenario AND the Half-A (unfenced) scenario, recording
for each worker an ordered timeline of: model returns, thaw, tool executions,
saver.put, saver.put_writes.  Answers:
  * is put_writes A's first post-thaw action?
  * does A ever execute tc_A2's tool in the fenced run?
  * do BOTH workers issue put_writes for the same (checkpoint_id, task_id)?

usage: write_tap.py                # orchestrator
       write_tap.py <json cfg>     # worker

Provenance: copied VERBATIM (not one character changed) from the adversarial-
review scratchpad as `e3_tap.py`. This is the put/put_writes write-tap
evidence behind "Half B passes for the right reason" in ../README.md -- see
that file for the full evidence-to-claim mapping. No path fix was needed:
this script already hardcodes this repo's absolute path.
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spike"))
sys.path.insert(0, str(ROOT / "proof"))

import psycopg  # noqa: E402
from _harness import ScriptedChatModel  # noqa: E402
from _harness import pg_conninfo  # noqa: E402
from deepagents import create_deep_agent  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402

from fence import (FencedPostgresSaver, claim, install_fence,  # noqa: E402
                   setup_database)

PY = str(ROOT / "spike" / ".venv" / "bin" / "python")
ME = str(pathlib.Path(__file__).resolve())
T0 = time.monotonic()
TAP: list[str] = []

SCRIPT_A = [{"content": "looking up k1", "tool": {"id": "tc_A1", "key": "k1"}},
            {"content": "worker A turn 2", "tool": {"id": "tc_A2", "key": "k2"}},
            {"content": "final answer from A", "tool": None}]
SCRIPT_B = [{"content": "worker B turn 2", "tool": {"id": "tc_B2", "key": "k2"}},
            {"content": "final answer from B", "tool": None}]


def rec(what: str) -> None:
    TAP.append(f"{(time.monotonic() - T0) * 1000:8.1f}ms  {what}")


@tool
def lookup(key: str) -> str:
    """Look up a key in the knowledge base."""
    rec(f"TOOL   lookup(key={key!r})  <-- side effect")
    return f"value-of-{key}"


def to_ai(spec):
    t = spec.get("tool")
    calls = [{"name": "lookup", "args": {"key": t["key"]}, "id": t["id"]}] if t else []
    return AIMessage(content=spec["content"], tool_calls=calls)


class FreezableModel(ScriptedChatModel):
    block_at: int = -1
    ready_file: str = ""
    thaw_file: str = ""

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if self.i == self.block_at:
            rec(f"MODEL  call #{self.i} -- announcing pid and BLOCKING (pre-SIGSTOP)")
            pathlib.Path(self.ready_file).write_text(str(os.getpid()))
            while not pathlib.Path(self.thaw_file).exists():
                time.sleep(0.02)
            rec("THAW   thaw file seen; resuming inside the model call")
        r = super()._generate(messages, stop, run_manager, **kw)
        ids = [c["id"] for c in (r.generations[0].message.tool_calls or [])]
        rec(f"MODEL  call #{self.i - 1} returned tool_calls={ids}")
        return r


def tapped(cls):
    class Tapped(cls):
        def put(self, config, checkpoint, metadata, new_versions):
            rec(f"PUT        cp=...{checkpoint['id'][-8:]} step={metadata.get('step')}")
            return super().put(config, checkpoint, metadata, new_versions)

        def put_writes(self, config, writes, task_id, task_path=""):
            rec(f"PUT_WRITES cp=...{config['configurable']['checkpoint_id'][-8:]}"
                f" task={task_id[:8]} channels={[w[0] for w in writes]}")
            return super().put_writes(config, writes, task_id, task_path)
    return Tapped


def worker(cfg) -> int:
    out = {"role": cfg["role"], "pid": os.getpid(), "ok": False}
    try:
        if cfg["role"] == "B" and cfg.get("fence") is not None:
            out["claimed_fence"] = claim(cfg["conninfo"], cfg["thread_id"],
                                        cfg["fence"] - 1, "B")
            cfg["fence"] = out["claimed_fence"]
        model = FreezableModel(responses=[to_ai(s) for s in cfg["script"]],
                              block_at=cfg.get("block_at", -1),
                              ready_file=cfg.get("ready_file", ""),
                              thaw_file=cfg.get("thaw_file", ""))
        conn = psycopg.connect(cfg["conninfo"], autocommit=True)
        if cfg.get("fence") is None:
            saver = tapped(PostgresSaver)(conn)
        else:
            saver = tapped(FencedPostgresSaver)(conn, cfg["thread_id"], cfg["fence"])
        agent = create_deep_agent(model=model, tools=[lookup], checkpointer=saver)
        rc = {"configurable": {"thread_id": cfg["thread_id"]}}
        inp = None if cfg["role"] == "B" else {"messages": [HumanMessage("go")]}
        agent.invoke(inp, rc, durability="sync")
        out["ok"] = True
    except BaseException as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}".strip()[:200]
        out["sqlstate"] = getattr(e, "sqlstate", None)
        tb = traceback.extract_tb(e.__traceback__)
        out["frames"] = [f"{pathlib.Path(f.filename).name}:{f.lineno} {f.name}"
                         for f in tb[-8:]]
        rec(f"RAISE  {out['error']}")
    out["tap"] = TAP
    pathlib.Path(cfg["out_file"]).write_text(json.dumps(out))
    return 0 if out["ok"] else 1


def run(conninfo, tid, fenced):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="e3-"))
    common = {"conninfo": conninfo, "thread_id": tid}
    a = dict(common, role="A", script=SCRIPT_A, block_at=1,
             ready_file=str(tmp / "ready"), thaw_file=str(tmp / "thaw"),
             out_file=str(tmp / "a.json"), fence=1 if fenced else None)
    b = dict(common, role="B", script=SCRIPT_B, block_at=-1,
             out_file=str(tmp / "b.json"), fence=2 if fenced else None)
    pa = subprocess.Popen([PY, ME, json.dumps(a)], stderr=subprocess.DEVNULL)
    end = time.time() + 90
    while not (tmp / "ready").exists() and time.time() < end:
        time.sleep(0.02)
    pid = int((tmp / "ready").read_text())
    os.kill(pid, signal.SIGSTOP)
    pb = subprocess.Popen([PY, ME, json.dumps(b)], stderr=subprocess.DEVNULL)
    pb.wait(timeout=240)
    (tmp / "thaw").write_text("go")
    os.kill(pid, signal.SIGCONT)
    try:
        pa.wait(timeout=240)
    except subprocess.TimeoutExpired:
        pa.kill()
    return (json.loads((tmp / "a.json").read_text()),
            json.loads((tmp / "b.json").read_text()))


def main():
    conninfo = pg_conninfo("e3_tap")
    setup_database(conninfo)
    for fenced in (True, False):
        tid = "e3-fenced" if fenced else "e3-unfenced"
        if fenced:
            install_fence(conninfo, tid, "A")
        a, b = run(conninfo, tid, fenced)
        print("=" * 78)
        print(f"{'FENCED (Half B)' if fenced else 'UNFENCED (Half A)'}"
              f"   A.ok={a['ok']} err={a.get('error')} sqlstate={a.get('sqlstate')}")
        print("=" * 78)
        for who, o in (("B", b), ("A", a)):
            print(f"  --- worker {who} timeline (ok={o['ok']}) ---")
            for line in o["tap"]:
                print(f"    {line}")
            if o.get("frames"):
                print("    raise frames (innermost last):")
                for f in o["frames"]:
                    print(f"      {f}")
        # did both issue put_writes for the same (cp, task)?
        def pws(o):
            return {ln.split("cp=")[1].split(" task=")[0] + "/" +
                    ln.split("task=")[1].split(" ")[0]
                    for ln in o["tap"] if "PUT_WRITES" in ln}
        both = pws(a) & pws(b)
        print(f"  (cp,task) pairs BOTH workers issued put_writes for: {sorted(both)}")
        print(f"  A executed lookup for keys: "
              f"{[ln.split('key=')[1].split(')')[0] for ln in a['tap'] if 'TOOL' in ln]}")
        thaw_idx = [i for i, ln in enumerate(a["tap"]) if "THAW" in ln]
        if thaw_idx:
            print(f"  A's actions after THAW: "
                  f"{[ln.split('ms  ')[1][:34] for ln in a['tap'][thaw_idx[0] + 1:]]}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(worker(json.loads(sys.argv[1])))
    main()
