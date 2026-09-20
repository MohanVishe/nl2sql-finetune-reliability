"""The analysis turns attempts into the headline; these check it on data with known answers.

A bug here would not crash anything -- it would quietly produce a plausible number. So the
estimators are checked against hand-computable cases, the bootstrap against a difference whose
true value is known, and the figures against being well-formed SVG.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import xml.dom.minidom
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyse = load("analyse")
make_charts = load("make_charts")


def write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def attempts(question_id: int, flags: list[bool], **extra) -> list[dict]:
    return [{"question_id": question_id, "run": i, "match": f, "match_strict": f, **extra}
            for i, f in enumerate(flags)]


def test_read_skips_questions_whose_gold_query_failed(tmp_path):
    rows = attempts(1, [True, False]) + attempts(2, [False, False], gold_failed=True)
    flags = analyse.read(write(tmp_path / "a.jsonl", rows), strict=False)
    assert flags == {1: [True, False]}


def test_align_uses_common_questions_and_the_shallowest_depth(tmp_path):
    a = analyse.read(write(tmp_path / "a.jsonl", attempts(1, [True] * 3) + attempts(2, [True])),
                     strict=False)
    b = analyse.read(write(tmp_path / "b.jsonl", attempts(1, [True] * 2) + attempts(3, [True])),
                     strict=False)
    assert analyse.align({"a": a, "b": b}) == ([1], 2)


def test_suite_matches_hand_computed_estimators():
    """Two questions, 2 attempts each: one always right, one right once.

    pass@2 = mean(1, 1) = 1.0        -- the flaky one still has a right answer among the two
    pass^2 = mean(1, 0) = 0.5        -- and fails the moment both must be right
    """
    from nl2sql_reliability.metrics import pass_at_k, pass_hat_k

    arm = {1: [True, True], 2: [True, False]}
    assert analyse.suite(arm, [1, 2], 2, 2, pass_at_k) == pytest.approx(1.0)
    assert analyse.suite(arm, [1, 2], 2, 2, pass_hat_k) == pytest.approx(0.5)


def test_bootstrap_brackets_a_known_difference_and_is_deterministic():
    per_question = {q: 0.10 for q in range(200)}  # every question improved by exactly 10 points
    low, high = analyse.bootstrap(per_question, list(range(200)), seed=0, resamples=500)
    assert low == pytest.approx(0.10) and high == pytest.approx(0.10)

    mixed = {q: (0.2 if q % 2 else -0.1) for q in range(200)}
    first = analyse.bootstrap(mixed, list(range(200)), seed=0, resamples=500)
    assert analyse.bootstrap(mixed, list(range(200)), seed=0, resamples=500) == first
    assert first[0] < 0.05 < first[1]  # true mean, inside the interval


def test_bootstrap_interval_widens_when_questions_disagree():
    agree = {q: 0.05 for q in range(100)}
    disagree = {q: (0.55 if q % 2 else -0.45) for q in range(100)}  # same mean, more spread
    narrow = analyse.bootstrap(agree, list(range(100)), seed=0, resamples=500)
    wide = analyse.bootstrap(disagree, list(range(100)), seed=0, resamples=500)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_charts_are_well_formed_svg(tmp_path):
    summary = {
        "questions": 4, "attempts_per_question": 3,
        "curves": {name: [{"k": k, "pass_at_k": 0.4 + k / 100, "pass_hat_k": 0.2,
                           "gap": 0.2 + k / 100} for k in (1, 2, 3)]
                   for name in ("baseline", "treatment")},
        "moved": {"solid_both": 1, "never_both": 1, "became_solid": 1, "lost_solid": 1,
                  "improved": 2, "worsened": 1},
    }
    training = {"log_history": [{"step": s, "loss": 0.2 - s / 10_000} for s in (10, 20, 30)]
                + [{"step": s, "eval_loss": 0.15 + s / 10_000} for s in (10, 20, 30)]}

    make_charts.curves_chart(summary, tmp_path / "curves.svg")
    make_charts.moved_chart(summary, tmp_path / "moved.svg")
    make_charts.training_chart(training, tmp_path / "training.svg")
    for name in ("curves.svg", "moved.svg", "training.svg"):
        xml.dom.minidom.parse(str(tmp_path / name))  # raises if malformed
        assert "<svg" in (tmp_path / name).read_text(encoding="utf-8")
