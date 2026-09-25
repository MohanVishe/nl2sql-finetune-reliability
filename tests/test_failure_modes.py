"""The failure-mode split decides what counts as a silent failure, so it is checked directly.

The distinction that matters: a wrong answer the database rejected (loud, catchable) versus a
wrong answer it happily returned (silent, reaches the user). Misfiling one as the other would
invert the study's practical claim, and nothing else in the pipeline would notice.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import xml.dom.minidom
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


failure_modes = load("failure_modes")
make_charts = load("make_charts")


def write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def attempt(question_id: int, *, match: bool, exec_error=None, **extra) -> dict:
    return {"question_id": question_id, "match": match, "exec_error": exec_error, **extra}


def test_outcomes_separate_loud_failures_from_silent_ones(tmp_path):
    rows = [
        attempt(1, match=True),
        attempt(1, match=False, exec_error="no such column: foo"),
        attempt(1, match=False),                       # ran fine, returned the wrong rows
        attempt(1, match=False, timed_out=True),       # a timeout is still a visible failure
    ]
    assert failure_modes.read(write(tmp_path / "a.jsonl", rows)) == {
        1: ["correct", "crashed", "silent_wrong", "crashed"]
    }


def test_questions_whose_gold_query_failed_are_dropped(tmp_path):
    rows = [attempt(1, match=True), attempt(2, match=False, gold_failed=True)]
    assert set(failure_modes.read(write(tmp_path / "a.jsonl", rows))) == {1}


def test_rates_weight_every_question_equally():
    arm = {1: ["correct"] * 4, 2: ["crashed"] * 2 + ["silent_wrong"] * 2}
    rates = failure_modes.rates(arm, [1, 2], 4)
    assert rates == {"correct": 0.5, "crashed": 0.25, "silent_wrong": 0.25}


def test_consistency_splits_never_flaky_and_always():
    arm = {
        1: ["silent_wrong"] * 4,                        # never
        2: ["correct", "crashed", "correct", "crashed"],  # flaky
        3: ["correct"] * 4,                             # always
    }
    buckets = failure_modes.consistency(arm, [1, 2, 3], 4)
    assert (buckets["never"], buckets["flaky"], buckets["always"]) == (1, 1, 1)
    assert buckets["spread"] == {0: 1, 1: 0, 2: 1, 3: 0, 4: 1}


def test_difference_is_paired_and_signed_treatment_minus_baseline():
    baseline = {1: ["crashed"] * 4, 2: ["crashed"] * 4}
    treatment = {1: ["silent_wrong"] * 4, 2: ["crashed"] * 4}
    diff = failure_modes.differences(baseline, treatment, [1, 2], 4, seed=0)
    assert diff["crashed"]["delta"] == -0.5          # half the attempts stopped crashing
    assert diff["silent_wrong"]["delta"] == +0.5     # and became silent failures instead
    assert diff["correct"]["delta"] == 0.0


def test_new_charts_are_well_formed_svg(tmp_path):
    report = {
        "questions": 3, "attempts_per_question": 4,
        "rates": {"baseline": {"correct": 0.5, "crashed": 0.25, "silent_wrong": 0.25},
                  "treatment": {"correct": 0.6, "crashed": 0.1, "silent_wrong": 0.3}},
        "consistency": {"baseline": {"never": 1, "flaky": 1, "always": 1},
                        "treatment": {"never": 0, "flaky": 1, "always": 2}},
        "differences": {"treatment_minus_baseline": {
            "correct": {"delta": 0.1}, "crashed": {"delta": -0.15},
            "silent_wrong": {"delta": 0.05}}},
    }
    for draw, name in ((make_charts.failure_modes_chart, "failure-modes.svg"),
                       (make_charts.consistency_chart, "consistency.svg")):
        out = tmp_path / name
        draw(report, out)
        xml.dom.minidom.parseString(out.read_text(encoding="utf-8"))
        assert "<svg" in out.read_text(encoding="utf-8")
    note = (tmp_path / "failure-modes.svg").read_text(encoding="utf-8")
    assert "crashes fell 15.0 points; right answers rose 10.0, silent wrong answers 5.0" in note
