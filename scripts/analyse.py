"""Turn raw attempts into the study's numbers, on the questions both arms answered.

    uv run python scripts/analyse.py --baseline results/p3-f0-base.jsonl \
        --treatment results/p3-f1-qlora.jsonl --out results/summary.json

Estimators come from P1's `nl2sql_reliability.metrics` -- the same code that produced the
published baseline -- so nothing is re-derived here:

    pass@k   the chance that at least one of k attempts is right   (capability)
    pass^k   the chance that all k attempts are right              (reliability)
    gap      pass@k - pass^k                                       (how flaky it is)

The headline is **delta-gap = gap(treatment) - gap(baseline)**, with a paired bootstrap
interval. The bootstrap resamples *questions*, not attempts: questions are the independent
units, and resampling attempts inside a question would give an interval far too narrow.
Pairing matters too -- both arms answered the same questions, so each resample must take both
arms' results for the same drawn questions.

Everything is computed on the intersection: questions both arms answered, with the same number
of repetitions. Subtracting two separately-computed headline numbers would silently compare
different question sets.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from nl2sql_finetune.stats import RESAMPLES, bootstrap


def read(path: Path, *, strict: bool) -> dict[int, list[bool]]:
    """question id -> one flag per attempt, in file order."""
    flags: dict[int, list[bool]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("gold_failed"):
                continue  # never sent to the model; P1 excludes these from scoring too
            key = "match_strict" if strict else "match"
            flags[int(row["question_id"])].append(bool(row[key]))
    return dict(flags)


def align(arms: dict[str, dict[int, list[bool]]]) -> tuple[list[int], int]:
    common = set.intersection(*(set(a) for a in arms.values()))
    depth = min(len(flags) for arm in arms.values() for qid, flags in arm.items()
                if qid in common)
    return sorted(common), depth


def suite(arm: dict[int, list[bool]], questions: list[int], depth: int, k: int,
          estimator) -> float:
    return statistics.fmean(
        estimator(depth, sum(arm[q][:depth]), k) for q in questions
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=Path("results/p3-f0-base.jsonl"))
    parser.add_argument("--treatment", type=Path, default=Path("results/p3-f1-qlora.jsonl"))
    parser.add_argument("--published", type=Path, help="P1's arm C, for the control row")
    parser.add_argument("--out", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--strict", action="store_true", help="score without column leniency")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from nl2sql_reliability.metrics import pass_at_k, pass_hat_k

    arms = {"baseline": read(args.baseline, strict=args.strict),
            "treatment": read(args.treatment, strict=args.strict)}
    if args.published:
        arms["published"] = read(args.published, strict=args.strict)
    questions, depth = align(arms)
    print(f"{len(questions)} questions answered by every arm, {depth} attempts each")

    curves: dict[str, list[dict]] = {}
    for name, arm in arms.items():
        rows = []
        for k in range(1, depth + 1):
            at_k = suite(arm, questions, depth, k, pass_at_k)
            hat_k = suite(arm, questions, depth, k, pass_hat_k)
            rows.append({"k": k, "pass_at_k": at_k, "pass_hat_k": hat_k, "gap": at_k - hat_k})
        curves[name] = rows

    # Per-question differences at full depth, then one interval per quantity.
    k = depth
    differences = {}
    for label, estimator in (("pass_at_k", pass_at_k), ("pass_hat_k", pass_hat_k)):
        per_question = {q: estimator(depth, sum(arms["treatment"][q][:depth]), k)
                        - estimator(depth, sum(arms["baseline"][q][:depth]), k)
                        for q in questions}
        low, high = bootstrap(per_question, questions, seed=args.seed)
        differences[label] = {"delta": statistics.fmean(per_question.values()),
                              "ci_low": low, "ci_high": high}
    gap_per_question = {
        q: (pass_at_k(depth, sum(arms["treatment"][q][:depth]), k)
            - pass_hat_k(depth, sum(arms["treatment"][q][:depth]), k))
        - (pass_at_k(depth, sum(arms["baseline"][q][:depth]), k)
           - pass_hat_k(depth, sum(arms["baseline"][q][:depth]), k))
        for q in questions}
    low, high = bootstrap(gap_per_question, questions, seed=args.seed)
    differences["gap"] = {"delta": statistics.fmean(gap_per_question.values()),
                          "ci_low": low, "ci_high": high}

    # What moved, per question: correct-attempt counts out of `depth`.
    moved = []
    for q in questions:
        before, after = sum(arms["baseline"][q][:depth]), sum(arms["treatment"][q][:depth])
        if before != after:
            moved.append({"question_id": q, "baseline": before, "treatment": after})
    buckets = {
        "solid_both": sum(1 for q in questions if sum(arms["baseline"][q][:depth]) == depth
                          and sum(arms["treatment"][q][:depth]) == depth),
        "never_both": sum(1 for q in questions if sum(arms["baseline"][q][:depth]) == 0
                          and sum(arms["treatment"][q][:depth]) == 0),
        "became_solid": sum(1 for q in questions if sum(arms["baseline"][q][:depth]) < depth
                            and sum(arms["treatment"][q][:depth]) == depth),
        "lost_solid": sum(1 for q in questions if sum(arms["baseline"][q][:depth]) == depth
                          and sum(arms["treatment"][q][:depth]) < depth),
        "improved": sum(1 for m in moved if m["treatment"] > m["baseline"]),
        "worsened": sum(1 for m in moved if m["treatment"] < m["baseline"]),
    }

    summary = {
        "questions": len(questions), "attempts_per_question": depth,
        "strict": args.strict, "resamples": RESAMPLES, "seed": args.seed,
        "arms": {name: str(path) for name, path in
                 (("baseline", args.baseline), ("treatment", args.treatment),
                  ("published", args.published)) if path},
        "curves": curves,
        "differences_at_full_k": differences,
        "moved": buckets,
        "moved_questions": moved,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'arm':<12}{'pass@k':>9}{'pass^k':>9}{'gap':>9}   (k={depth})")
    for name, rows in curves.items():
        row = rows[-1]
        print(f"{name:<12}{row['pass_at_k']:>8.1%}{row['pass_hat_k']:>8.1%}{row['gap']:>8.1%}")
    print(f"\ntreatment - baseline at k={depth}:")
    for label, d in differences.items():
        zero = "  (includes zero)" if d["ci_low"] <= 0 <= d["ci_high"] else "  (excludes zero)"
        print(f"  {label:<10}{d['delta']:>+7.1%}  95% CI [{d['ci_low']:>+6.1%},"
              f"{d['ci_high']:>+6.1%}]{zero}")
    print(f"\nmoved: {buckets}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
