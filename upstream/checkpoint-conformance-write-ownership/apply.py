#!/usr/bin/env python3
"""Add the WRITE_OWNERSHIP capability to a `checkpoint-conformance` source tree.

    python3 apply.py <path-to-libs/checkpoint-conformance>

Why a script and not just the patch. The suite gains capabilities: `delta_channel_history`
landed after 0.0.2 was published, and every one of the four files this touches is a list
that grew. A patch pinned to one revision goes stale on contact with the next one, and the
resulting conflict is in exactly the kind of enum-and-dict boilerplate where a hand
resolution silently drops an entry.

So the four mechanical edits are expressed as *append to the end of the relevant block*,
which is stable across anything upstream is likely to do, and the checked-in patches are
generated from this script rather than maintained beside it.

Idempotent: running twice is a no-op, so it is safe to re-run after a rebase.

The two substantive files -- `ownership.py` and `spec/test_write_ownership.py` -- are
copied verbatim from `tree/`. Nothing about them is generated.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE / "tree"

CAP_NAME = "WRITE_OWNERSHIP"
CAP_VALUE = "write_ownership"
CAP_METHOD = "aclaim_write_ownership"
RUNNER = "run_write_ownership_tests"

_METHOD_MAP_COMMENT = """\
    # Absent from BaseCheckpointSaver entirely -- which is why adding this capability
    # needs no base-class change. `_is_overridden` below returns True for a method the
    # base class does not define, so a saver is detected purely by defining it. See
    # ownership.py.
"""


def _append_to_block(text: str, *, open_pat: str, entry: str, close: str, label: str) -> str:
    """Insert *entry* immediately before the line that closes a block.

    `open_pat` locates the block's start; `close` is the first line at the block's own
    indentation that ends it. Deliberately not an "insert after member X" match: X is
    exactly what a new upstream capability displaces.
    """
    m = re.search(open_pat, text)
    if m is None:
        raise SystemExit(f"could not locate {label} (pattern: {open_pat!r})")
    end = text.index(close, m.end())
    return text[:end] + entry + text[end:]


def patch_capabilities(path: Path) -> None:
    s = path.read_text()
    if CAP_NAME in s:
        return
    s = _append_to_block(
        s,
        open_pat=r"class Capability\(str, Enum\):",
        entry=f'    {CAP_NAME} = "{CAP_VALUE}"\n',
        close="\n\n# Capabilities that every checkpointer must support.",
        label="the Capability enum",
    )
    s = _append_to_block(
        s,
        open_pat=r"EXTENDED_CAPABILITIES = frozenset\(\n    \{\n",
        entry=f"        Capability.{CAP_NAME},\n",
        close="    }\n)\n\nALL_CAPABILITIES",
        label="EXTENDED_CAPABILITIES",
    )
    s = _append_to_block(
        s,
        open_pat=r"_CAPABILITY_METHOD_MAP: dict\[Capability, str\] = \{\n",
        entry=_METHOD_MAP_COMMENT + f'    Capability.{CAP_NAME}: "{CAP_METHOD}",\n',
        close="}\n\n\n@dataclass(frozen=True)",
        label="_CAPABILITY_METHOD_MAP",
    )
    path.write_text(s)


def patch_validate(path: Path) -> None:
    s = path.read_text()
    if RUNNER in s:
        return
    s = _append_to_block(
        s,
        open_pat=r"from langgraph\.checkpoint\.conformance\.spec\.",
        entry=(
            "from langgraph.checkpoint.conformance.spec.test_write_ownership import (\n"
            f"    {RUNNER},\n"
            ")\n"
        ),
        close="\n# Maps capability to its runner function.",
        label="the spec imports in validate.py",
    )
    s = _append_to_block(
        s,
        open_pat=r"_RUNNERS = \{\n",
        entry=f"    Capability.{CAP_NAME}: {RUNNER},\n",
        close="}\n\n\nasync def validate(",
        label="_RUNNERS",
    )
    path.write_text(s)


def patch_spec_init(path: Path) -> None:
    s = path.read_text()
    if RUNNER in s:
        return
    s = _append_to_block(
        s,
        open_pat=r"from langgraph\.checkpoint\.conformance\.spec\.",
        entry=(
            "from langgraph.checkpoint.conformance.spec.test_write_ownership import (\n"
            f"    {RUNNER},\n"
            ")\n"
        ),
        close="\n__all__ = [",
        label="the spec imports in spec/__init__.py",
    )
    s = _append_to_block(
        s,
        open_pat=r"__all__ = \[\n",
        entry=f'    "{RUNNER}",\n',
        close="]\n",
        label="spec __all__",
    )
    path.write_text(s)


def patch_package_init(path: Path) -> None:
    s = path.read_text()
    if "StaleWriteOwnerError" in s:
        return
    s = s.replace(
        "from langgraph.checkpoint.conformance.initializer import checkpointer_test\n",
        "from langgraph.checkpoint.conformance.initializer import checkpointer_test\n"
        "from langgraph.checkpoint.conformance.ownership import (\n"
        "    StaleWriteOwnerError,\n"
        "    is_stale_write_owner_rejection,\n"
        ")\n",
        1,
    )
    s = _append_to_block(
        s,
        open_pat=r"__all__ = \[\n",
        entry='    "StaleWriteOwnerError",\n    "is_stale_write_owner_rejection",\n',
        close="]\n",
        label="package __all__",
    )
    path.write_text(s)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    root = Path(argv[1]).resolve()
    conf = root / "langgraph" / "checkpoint" / "conformance"
    if not conf.is_dir():
        raise SystemExit(f"not a checkpoint-conformance tree: {root} (no {conf})")

    for rel in (
        "langgraph/checkpoint/conformance/ownership.py",
        "langgraph/checkpoint/conformance/spec/test_write_ownership.py",
        "tests/test_write_ownership.py",
    ):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TREE / rel, dst)

    patch_capabilities(conf / "capabilities.py")
    patch_validate(conf / "validate.py")
    patch_spec_init(conf / "spec" / "__init__.py")
    patch_package_init(conf / "__init__.py")

    print(f"WRITE_OWNERSHIP applied to {root}")
    print("  run:  pytest tests/test_write_ownership.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
