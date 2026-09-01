"""Shared spike harness. Pinned versions recorded in VERSIONS.
Import from any spike script:  from _harness import pg_conninfo, ScriptedChatModel
"""
from __future__ import annotations
import pathlib, uuid
from typing import Any, Iterator, Sequence

SPIKE = pathlib.Path(__file__).parent.absolute()

# ---------------------------------------------------------------- postgres ---
def _server():
    import pgserver
    return pgserver.get_server(str(SPIKE / "pgdata"), cleanup_mode=None)

def pg_conninfo(dbname: str | None = None) -> str:
    """Start (or reuse) the embedded Postgres and return a psycopg conninfo
    string for a freshly-created database. Each caller should use its own db."""
    db = _server()
    name = dbname or ("spike_" + uuid.uuid4().hex[:10])
    import psycopg
    with psycopg.connect(db.get_uri(), autocommit=True) as c:
        exists = c.execute("select 1 from pg_database where datname=%s", (name,)).fetchone()
        if not exists:
            c.execute(f'create database "{name}"')
    base = db.get_uri()
    return base.replace("/postgres?", f"/{name}?")

# ------------------------------------------------------------- fake model ---
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class ScriptedChatModel(BaseChatModel):
    """Deterministic chat model: returns scripted AIMessages in order, so graphs
    and agents run with no API key and no nondeterminism. Accepts bind_tools."""
    responses: list[AIMessage] = []
    i: int = 0
    loop: bool = False

    @property
    def _llm_type(self) -> str: return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kw: Any):  # noqa: D102
        return self

    def with_structured_output(self, schema: Any, **kw: Any):  # noqa: D102
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        if self.i >= len(self.responses):
            if self.loop and self.responses:
                self.i = 0
            else:
                msg = AIMessage(content="done")
                return ChatResult(generations=[ChatGeneration(message=msg)])
        msg = self.responses[self.i]
        self.i += 1
        # copy so repeated runs (replay) do not mutate shared state
        return ChatResult(generations=[ChatGeneration(message=msg.model_copy())])

def scripted(*responses: AIMessage, loop: bool = False) -> ScriptedChatModel:
    return ScriptedChatModel(responses=list(responses), loop=loop)
