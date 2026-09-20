"""Import a Q4_K_M GGUF into Ollama exactly the way the baseline was served.

    uv run python scripts/build_ollama_model.py --gguf ../gguf/f0-base.Q4_K_M.gguf \
        --name p3-f0-base

The GGUF comes from `to_gguf.py` (llama.cpp): Ollama's own `--quantize` cannot produce Q4_K_M,
so the weights arrive already quantised and are imported as-is.

The Modelfile's TEMPLATE, SYSTEM, PARAMETER and LICENSE are **copied programmatically** from the
official `qwen2.5-coder:3b` -- the model P1 measured -- never retyped. After `ollama create`,
the new model is inspected and must report the same quantisation, the same template and the
same system prompt, or the script fails. A model served with a different template would be
evaluated on a format it was not trained on, and the comparison would measure the format.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REFERENCE = "qwen2.5-coder:3b"
HOST = "http://localhost:11434"


def show(model: str) -> dict:
    request = urllib.request.Request(
        f"{HOST}/api/show", data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def modelfile_for(gguf: Path, reference: dict) -> str:
    """FROM the new weights; everything else inherited verbatim from the reference model."""
    lines = [f"FROM {gguf.resolve().as_posix()}"]
    lines.append(f'TEMPLATE """{reference["template"]}"""')
    if reference.get("system"):
        lines.append(f'SYSTEM """{reference["system"]}"""')
    for line in (reference.get("parameters") or "").splitlines():
        key, _, value = line.strip().partition(" ")
        if key:
            lines.append(f"PARAMETER {key} {value.strip()}")
    # The Qwen Research License travels with any derivative.
    if reference.get("license"):
        lines.append(f'LICENSE """{reference["license"]}"""')
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", type=Path, required=True, help="Q4_K_M GGUF from to_gguf.py")
    parser.add_argument("--name", required=True, help="Ollama model name to create")
    args = parser.parse_args()

    reference = show(REFERENCE)
    modelfile = modelfile_for(args.gguf, reference)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Modelfile"
        # newline="\n": on Windows the default translation writes \r\n, and Ollama keeps the
        # carriage returns inside the TEMPLATE -- every prompt would then differ from the
        # baseline's. The verification below caught exactly that.
        path.write_text(modelfile, encoding="utf-8", newline="\n")
        print(modelfile.split("\n", 1)[0], "+ template/system/parameters/license from", REFERENCE)
        completed = subprocess.run(["ollama", "create", args.name, "-f", str(path)], check=False)
    if completed.returncode:
        print(f"error: ollama create exited {completed.returncode}", file=sys.stderr)
        return completed.returncode

    built = show(args.name)
    ref_level = reference.get("details", {}).get("quantization_level")
    new_level = built.get("details", {}).get("quantization_level")
    problems = []
    if new_level != ref_level:
        problems.append(f"quantisation {new_level!r} != reference {ref_level!r}")
    if built.get("template") != reference.get("template"):
        problems.append("template differs from the reference")
    if (built.get("system") or "") != (reference.get("system") or ""):
        problems.append("system prompt differs from the reference")
    if (built.get("parameters") or "") != (reference.get("parameters") or ""):
        problems.append("parameters differ from the reference")
    if built.get("details", {}).get("family") != reference.get("details", {}).get("family"):
        problems.append("architecture family differs")

    print(f"built {args.name}: {built.get('details')}")
    if problems:
        print("error: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"verified: quantisation {new_level}, template, system and parameters identical "
          f"to {REFERENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
