"""How a file path is written into a published result.

Results are read on other machines, so a recorded path must not depend on where this one keeps
its checkouts. A path inside the working directory is recorded relative to it. A path inside
some other git checkout is recorded as `<repo>/<path in repo>` (e.g.
`nl2sql-reliability/results/final/local-7b-single.jsonl`), whichever way it was given. Anything
else keeps the relative spelling it was given, or -- if it was given as an absolute path -- just
its file name (e.g. an Ollama weights blob, whose name is its sha256).
"""

from __future__ import annotations

from pathlib import Path


def portable(path: Path | str) -> str:
    given = Path(path)
    resolved = given.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        pass
    for parent in resolved.parents:
        if (parent / ".git").exists():
            return f"{parent.name}/{resolved.relative_to(parent).as_posix()}"
    return resolved.name if given.is_absolute() else given.as_posix()
