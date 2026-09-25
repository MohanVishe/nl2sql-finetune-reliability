"""Evaluate one Ollama model through P1's harness, unchanged, at P1's settings.

    uv run python scripts/evaluate.py --arm p3-f0-base --model p3-f0-base
    uv run python scripts/evaluate.py --arm p3-f1-qlora --model p3-f1-qlora

This is a thin, checking wrapper around P1's own `scripts/run_arm.py`: it does not generate,
execute or score anything itself. Before starting it insists that

- the P1 clone is at the pinned commit with a clean tree (the harness *is* the measurement), and
- the model's template, system prompt and parameters equal `qwen2.5-coder:3b`'s.

It records the Ollama server version, P1 commit and exact command beside the results
(`results/<arm>.run.json`). Arms are only compared when their Ollama versions match:
Ollama updates itself, and P1's baseline (arm C) ran on 0.34.1.

Temperature (0.2), num_ctx (8192), max new tokens (512) and k (10) are P1's defaults and are
deliberately not exposed here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from nl2sql_finetune.paths import portable

P1_COMMIT = "a0ea7fbff713860d2f1c44c039229b8849e5ca5d"
DEFAULT_P1 = Path("../nl2sql-reliability")
HOST = "http://localhost:11434"
REFERENCE = "qwen2.5-coder:3b"


def git(p1: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(p1), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def api(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(f"{HOST}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--p1-dir", type=Path, default=DEFAULT_P1)
    parser.add_argument("--questions", type=int, default=0, help="pilot subset; 0 = all 498")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.p1_dir = args.p1_dir.resolve()  # the harness runs with cwd=p1_dir

    head = git(args.p1_dir, "rev-parse", "HEAD")
    if head != P1_COMMIT:
        raise SystemExit(f"error: P1 is at {head}, pinned {P1_COMMIT}")
    if git(args.p1_dir, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("error: P1 clone has local modifications")

    reference = api("/api/show", {"model": REFERENCE})
    model = api("/api/show", {"model": args.model})
    for field in ("template", "system", "parameters"):
        if (model.get(field) or "") != (reference.get(field) or ""):
            raise SystemExit(f"error: {args.model} {field} differs from {REFERENCE}")
    if model["details"].get("quantization_level") != reference["details"].get("quantization_level"):
        raise SystemExit("error: quantisation differs from the reference")

    args.results.mkdir(parents=True, exist_ok=True)
    out = (args.results / f"{args.arm}.jsonl").resolve()
    command = [str(args.p1_dir / ".venv" / "Scripts" / "python.exe"), "scripts/run_arm.py",
               "--arm", args.arm, "--model", args.model, "--k", "10", "--out", str(out)]
    if args.questions:
        command += ["--questions", str(args.questions)]

    record = {
        "arm": args.arm, "model": args.model,
        "ollama_version": api("/api/version")["version"],
        "p1_commit": head, "command": [portable(part) if Path(part).is_absolute() else part
                                        for part in command],
        "model_details": model.get("details"),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    sidecar = args.results / f"{args.arm}.run.json"
    previous = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None
    if previous and previous["ollama_version"] != record["ollama_version"]:
        raise SystemExit(f"error: resuming {args.arm} on Ollama {record['ollama_version']}, "
                         f"but it started on {previous['ollama_version']}")
    sidecar.write_text(json.dumps(previous or record, indent=2), encoding="utf-8")

    print(f"{args.arm}: {args.model} on Ollama {record['ollama_version']}, P1 {head[:7]}")
    completed = subprocess.run(command, cwd=args.p1_dir, check=False)

    final = json.loads(sidecar.read_text(encoding="utf-8"))
    final["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    final["exit_code"] = completed.returncode
    sidecar.write_text(json.dumps(final, indent=2), encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
