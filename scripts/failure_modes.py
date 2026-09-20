"""What kind of wrong: crashes, silent wrong answers, and how flaky each arm is.

    uv run python scripts/failure_modes.py --out results/failure-modes.json

pass@k and pass^k score an attempt right or wrong. That hides two things a team shipping the
model would want to know.

**How it fails.** An attempt can fail loudly -- the SQL does not run, and the application gets
an exception it can catch -- or quietly, returning a perfectly well-formed table of the wrong
rows. Loud failures are cheap; quiet ones reach the user. Fine-tuning can trade one for the
other without moving accuracy much, and the headline metrics would not show it.

**Where the flakiness lives.** A reliability gap can mean every question is a coin flip, or it
can mean most questions are settled -- always right or always wrong -- and a fixed minority is
unstable. Those call for completely different fixes. Counting questions by how many of their
ten attempts succeeded separates them.

Both are read off the same attempt records the headline uses, with the same paired bootstrap
over questions, so the intervals are comparable.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from nl2sql_finetune.stats import RESAMPLES, bootstrap

OUTCOMES = ("correct", "crashed", "silent_wrong")


def read(path: Path) -> dict[int, list[str]]:
    """question id -> one outcome per attempt.

    `crashed` means the database refused the query. `silent_wrong` means it ran and returned
    the wrong rows -- the failure nothing downstream can detect.
    """
    outcomes: dict[int, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("gold_failed"):
                continue  # never sent to the model; P1 excludes these from scoring too
            if row.get("match"):
                outcome = "correct"
            elif row.get("exec_error") is not None or row.get("timed_out"):
                outcome = "crashed"
            else:
                outcome = "silent_wrong"
            outcomes[int(row["question_id"])].append(outcome)
    return dict(outcomes)


def align(arms: dict[str, dict[int, list[str]]]) -> tuple[list[int], int]:
    common = set.intersection(*(set(a) for a in arms.values()))
    depth = min(len(v) for arm in arms.values() for q, v in arm.items() if q in common)
    return sorted(common), depth


def rates(arm: dict[int, list[str]], questions: list[int], depth: int) -> dict[str, float]:
    """Per-attempt rate of each outcome, every question weighted equally."""
    return {
        outcome: statistics.fmean(
            arm[q][:depth].count(outcome) / depth for q in questions
        )
        for outcome in OUTCOMES
    }


def consistency(arm: dict[int, list[str]], questions: list[int], depth: int) -> dict[str, int]:
    """Questions split by how many of their attempts were right.

    `flaky` is the only bucket a reliability fix can act on: the model already reaches the
    answer sometimes. `never` is a capability ceiling, `always` is already solved.
    """
    buckets = {"never": 0, "flaky": 0, "always": 0}
    spread = dict.fromkeys(range(depth + 1), 0)
    for q in questions:
        hits = arm[q][:depth].count("correct")
        spread[hits] += 1
        buckets["never" if hits == 0 else "always" if hits == depth else "flaky"] += 1
    return {**buckets, "spread": spread}


def differences(baseline: dict[int, list[str]], treatment: dict[int, list[str]],
                questions: list[int], depth: int, *, seed: int) -> dict[str, dict]:
    """treatment - baseline for each outcome rate, with a paired interval."""
    out = {}
    for outcome in OUTCOMES:
        per_question = {
            q: treatment[q][:depth].count(outcome) / depth
               - baseline[q][:depth].count(outcome) / depth
            for q in questions
        }
        delta = statistics.fmean(per_question[q] for q in questions)
        low, high = bootstrap(per_question, questions, seed=seed)
        out[outcome] = {"delta": delta, "ci_low": low, "ci_high": high}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=Path("results/p3-f0-base.jsonl"))
    parser.add_argument("--treatment", type=Path, default=Path("results/p3-f1-qlora.jsonl"))
    parser.add_argument("--reference", type=Path,
                        help="a third arm to compare the treatment against, e.g. the 7B")
    parser.add_argument("--out", type=Path, default=Path("results/failure-modes.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arms = {"baseline": read(args.baseline), "treatment": read(args.treatment)}
    if args.reference:
        arms["reference"] = read(args.reference)
    questions, depth = align(arms)
    print(f"{len(questions)} questions answered by every arm, {depth} attempts each")

    report = {
        "questions": len(questions),
        "attempts_per_question": depth,
        "resamples": RESAMPLES,
        "seed": args.seed,
        "arms": {name: str(path) for name, path in
                 (("baseline", args.baseline), ("treatment", args.treatment),
                  ("reference", args.reference)) if path},
        "rates": {name: rates(arm, questions, depth) for name, arm in arms.items()},
        "consistency": {name: consistency(arm, questions, depth)
                        for name, arm in arms.items()},
        "differences": {
            "treatment_minus_baseline": differences(
                arms["baseline"], arms["treatment"], questions, depth, seed=args.seed),
        },
    }
    if "reference" in arms:
        report["differences"]["treatment_minus_reference"] = differences(
            arms["reference"], arms["treatment"], questions, depth, seed=args.seed)

    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    width = max(len(n) for n in arms)
    print(f"\n{'arm':<{width}}  {'correct':>9}{'crashed':>10}{'silent wrong':>14}"
          f"{'never':>8}{'flaky':>7}{'always':>8}")
    for name in arms:
        r, c = report["rates"][name], report["consistency"][name]
        print(f"{name:<{width}}  {r['correct']*100:>8.1f}%{r['crashed']*100:>9.1f}%"
              f"{r['silent_wrong']*100:>13.1f}%{c['never']:>8}{c['flaky']:>7}{c['always']:>8}")
    for label, diff in report["differences"].items():
        print(f"\n{label.replace('_', ' ')}, percentage points:")
        for outcome, d in diff.items():
            verdict = ("excludes zero" if d["ci_low"] * d["ci_high"] > 0 else "includes zero")
            print(f"  {outcome:<13}{d['delta']*100:+7.1f}  "
                  f"[{d['ci_low']*100:+6.1f}, {d['ci_high']*100:+6.1f}]  {verdict}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
