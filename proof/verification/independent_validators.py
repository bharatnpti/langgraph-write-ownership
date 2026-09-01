"""Attack 7: does anything in the installed packages independently agree that the
message list the forked thread reads back is invalid?

Reads the REAL durable state left by the tapped run (db da_tap, thread half-a).

Provenance: copied verbatim from the adversarial-review scratchpad as
`validate.py`. Only the sys.path setup below was changed, to point at this
repo's real spike/ and proof/ instead of the scratchpad's throwaway
`pristine/` copy of fence.py -- see ../README.md for the evidence-to-claim
mapping.

SETUP NOTE (read before running): main() below makes two load() calls.
  1. load("narrow-deep", "da_narrow_deep") -- populated by running this
     directory's tier_delta_vs_plain.py in "drive deep" mode first (it writes
     to that exact db/thread). Every probe that follows -- deepagents' own
     repair middleware, langchain_anthropic, langchain_core, google_genai,
     the anthropic SDK -- runs against THIS list, so it is the one that
     matters for the claim.
  2. load("half-b") (db defaults to "da_tap") -- this was populated, in the
     original review, by a plain run of proof.py pointed at a "da_tap"
     database (see ../README.md, tapped/logged). That populating script is
     not preserved standalone in this directory -- BUT, as verified when this
     file was added, the state itself is still there: the embedded Postgres
     cluster at spike/pgdata is long-lived across the whole review (the task
     that produced this directory was explicit about not reinitialising it),
     so this second load() currently reads back the real 8-message "half-b"
     thread from that same review session, not an empty list. If spike/pgdata
     is ever reinitialised, this probe (only) would go back to reading an
     empty list -- it feeds one contrastive show() print, not any of the
     probes below, so the script would still run to completion either way.
"""
from __future__ import annotations

import pathlib
import sys
import traceback

REPO = pathlib.Path(__file__).resolve().parent.parent.parent  # verification -> proof -> repo root
sys.path.insert(0, str(REPO / "spike"))
sys.path.insert(0, str(REPO / "proof"))

import psycopg  # noqa: E402
from _harness import ScriptedChatModel, pg_conninfo  # noqa: E402
from deepagents import create_deep_agent  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402

from fence import orphans  # noqa: E402


@tool
def lookup(key: str) -> str:
    """Look up a key in the knowledge base."""
    return f"value-of-{key}"


def load(thread: str, db: str = "da_tap"):
    conninfo = pg_conninfo(db)
    conn = psycopg.connect(conninfo, autocommit=True)
    agent = create_deep_agent(model=ScriptedChatModel(responses=[]), tools=[lookup],
                              checkpointer=PostgresSaver(conn))
    st = agent.get_state({"configurable": {"thread_id": thread}})
    return st.values.get("messages", [])


def show(msgs, label):
    print(f"\n=== {label}: {len(msgs)} messages ===")
    for m in msgs:
        print(f"  {type(m).__name__:<13} {str(m.content)[:24]:<26}"
              f" tool_calls={[c['id'] for c in (getattr(m, 'tool_calls', None) or [])] or '-'}"
              f" tool_call_id={getattr(m, 'tool_call_id', None) or '-'}")
    print(f"  proof's own orphans(): {orphans(msgs) or 'NONE'}")


def probe(name, fn):
    print(f"\n---- {name}")
    try:
        out = fn()
        print(f"  ACCEPTED (no complaint). result head: {str(out)[:400]}")
    except BaseException as e:  # noqa: BLE001
        print(f"  RAISED {type(e).__name__}: {str(e)[:300]}")
        print("  " + traceback.format_exc().strip().splitlines()[-2].strip())


def main() -> None:
    msgs = load("narrow-deep", "da_narrow_deep")
    show(msgs, "durable state of the FORKED thread (narrow-deep, fencing OFF, unprobed)")
    ok = load("half-b")
    show(ok, "durable state of the FENCED thread (half-b) for contrast")

    # 1. deepagents' own repair middleware
    print("\n---- deepagents PatchToolCallsMiddleware.before_agent (the framework's"
          " own repair of dangling tool calls)")
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    res = PatchToolCallsMiddleware().before_agent({"messages": msgs}, None)
    if res is None:
        print("  middleware returned None: it sees nothing to patch")
    else:
        patched = [m for m in res["messages"] if type(m).__name__ != "RemoveMessage"]
        print(f"  middleware rewrote the list -> {len(patched)} messages")
        for m in patched:
            print(f"    {type(m).__name__:<13} {str(m.content)[:44]:<46}"
                  f" tcid={getattr(m, 'tool_call_id', None) or '-'}")
        print(f"  orphans() AFTER the framework's own repair: {orphans(patched) or 'NONE'}")

    # 2. langchain_anthropic's request formatter
    from langchain_anthropic import ChatAnthropic
    from langchain_anthropic.chat_models import _format_messages
    probe("langchain_anthropic._format_messages(msgs)", lambda: _format_messages(msgs))
    llm = ChatAnthropic(model="claude-sonnet-4-5", api_key="not-a-real-key")
    probe("ChatAnthropic._get_request_payload(msgs)",
          lambda: llm._get_request_payload(msgs))

    # 3. langchain_core's OpenAI-format converter
    from langchain_core.messages.utils import convert_to_openai_messages
    probe("langchain_core convert_to_openai_messages(msgs)",
          lambda: convert_to_openai_messages(msgs))

    # 4. google genai formatter
    try:
        from langchain_google_genai._function_utils import (  # noqa: F401
            _ToolConfigDict,
        )
        from langchain_google_genai.chat_models import _parse_chat_history
        probe("langchain_google_genai._parse_chat_history(msgs)",
              lambda: _parse_chat_history(msgs))
    except Exception as e:  # noqa: BLE001
        print(f"\n---- langchain_google_genai probe unavailable: {e}")

    # 5. anthropic SDK client-side validation of the same body
    body = llm._get_request_payload(msgs) if True else None
    print("\n---- anthropic SDK: does it validate the message list client-side?")
    try:
        import anthropic.types as at
        params = {k: v for k, v in body.items() if k in
                  ("model", "messages", "max_tokens", "system", "tools")}
        params.setdefault("max_tokens", 16)
        at.MessageCreateParams  # exists?
        print("  anthropic.types.MessageCreateParams is a TypedDict:"
              f" {type(at.MessageCreateParams).__name__} -> TypedDicts do no runtime"
              " validation, so the SDK cannot reject this body locally.")
        print(f"  message roles in the payload: "
              f"{[m.get('role') for m in params['messages']]}")
        for m in params["messages"]:
            blocks = m["content"] if isinstance(m["content"], list) else [m["content"]]
            kinds = [b.get("type") if isinstance(b, dict) else "text" for b in blocks]
            ids = [b.get("id") or b.get("tool_use_id") for b in blocks
                   if isinstance(b, dict)]
            print(f"    role={m.get('role'):<9} blocks={kinds} ids={ids}")
    except BaseException as e:  # noqa: BLE001
        print(f"  probe failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
