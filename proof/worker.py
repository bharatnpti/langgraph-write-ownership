"""One worker process. Runs a real Deep Agents loop against one thread_id.

Spawned twice per half by proof.py. Worker A starts the run and freezes inside
its second model call; worker B takes the thread over with invoke(None, config).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "spike"))

import psycopg  # noqa: E402
from _harness import ScriptedChatModel  # noqa: E402
from deepagents import create_deep_agent  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402

from fence import FencedPostgresSaver, claim  # noqa: E402


@tool
def lookup(key: str) -> str:
    """Look up a key in the knowledge base."""
    return f"value-of-{key}"


class FreezableModel(ScriptedChatModel):
    """Scripted model that announces itself and blocks on a file before
    returning response `block_at`. SIGSTOP then freezes the poll loop
    mid-super-step: a worker that is not dead, only unreachable."""

    block_at: int = -1
    ready_file: str = ""
    thaw_file: str = ""

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if self.i == self.block_at:
            pathlib.Path(self.ready_file).write_text(str(os.getpid()))
            while not pathlib.Path(self.thaw_file).exists():
                time.sleep(0.02)
        return super()._generate(messages, stop, run_manager, **kw)


def to_ai(spec: dict) -> AIMessage:
    t = spec.get("tool")
    calls = [{"name": "lookup", "args": {"key": t["key"]}, "id": t["id"]}] if t else []
    return AIMessage(content=spec["content"], tool_calls=calls)


def main() -> int:
    cfg = json.loads(sys.argv[1])
    out: dict = {"role": cfg["role"], "pid": os.getpid(), "ok": False}
    try:
        if cfg["role"] == "B" and cfg.get("fence") is not None:
            out["claimed_fence"] = claim(
                cfg["conninfo"], cfg["thread_id"], cfg["fence"] - 1, "B"
            )
            cfg["fence"] = out["claimed_fence"]

        model = FreezableModel(
            responses=[to_ai(s) for s in cfg["script"]],
            block_at=cfg.get("block_at", -1),
            ready_file=cfg.get("ready_file", ""),
            thaw_file=cfg.get("thaw_file", ""),
        )
        conn = psycopg.connect(cfg["conninfo"], autocommit=True)
        if cfg.get("fence") is None:
            saver = PostgresSaver(conn)
        else:
            saver = FencedPostgresSaver(conn, cfg["thread_id"], cfg["fence"])
        if cfg.get("force_no_pipeline"):
            saver.supports_pipeline = False

        agent = create_deep_agent(model=model, tools=[lookup], checkpointer=saver)
        rc = {"configurable": {"thread_id": cfg["thread_id"]}}
        # The resume path writes NOTHING to the thread before invoke(None, ...):
        # no update_state, no fork (that renames a subagent's checkpoint_ns).
        inp = None if cfg["role"] == "B" else {"messages": [HumanMessage("go")]}
        res = agent.invoke(inp, rc, durability="sync")
        out["ok"] = True
        out["messages"] = [
            [type(m).__name__, str(m.content)[:40],
             [c["id"] for c in (getattr(m, "tool_calls", None) or [])],
             getattr(m, "tool_call_id", None)]
            for m in res["messages"]
        ]
    except BaseException as e:  # noqa: BLE001 - report, never mask
        out["error"] = f"{type(e).__name__}: {e}".strip()[:300]
        out["sqlstate"] = getattr(e, "sqlstate", None)
        out["traceback_tail"] = traceback.format_exc().strip().splitlines()[-1]
    pathlib.Path(cfg["out_file"]).write_text(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
