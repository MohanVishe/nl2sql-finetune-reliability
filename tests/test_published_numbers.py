"""The counts the README states are recomputed here from the committed result files.

Every figure below appears in README.md or docs/EXPLAINED.md. A wording pass once got several
of them wrong (190 for 150, "a fifth" for about a third, 60M for 30M), so they are checked
against the raw attempts rather than trusted. The 7B's raw attempts live in the earlier
project; its figures are read from the committed comparison reports instead.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from nl2sql_finetune.paths import portable

RESULTS = Path(__file__).resolve().parents[1] / "results"


def hits(name: str) -> dict[int, int]:
    """question id -> right attempts out of ten, scored questions only."""
    out: Counter[int] = Counter()
    seen: set[int] = set()
    with open(RESULTS / name, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["gold_failed"]:
                continue
            seen.add(row["question_id"])
            out[row["question_id"]] += bool(row["match"])
    return {q: out[q] for q in seen}


def band(right: int) -> str:
    return "never" if right == 0 else "always" if right == 10 else "partly"


@pytest.fixture(scope="module")
def arms() -> tuple[dict[int, int], dict[int, int]]:
    return hits("p3-f0-base.jsonl"), hits("p3-f1-qlora.jsonl")


def test_rows_published_and_scored():
    for name in ("p3-f0-base.jsonl", "p3-f1-qlora.jsonl"):
        rows = [json.loads(line) for line in open(RESULTS / name, encoding="utf-8")]
        assert len(rows) == 4980
        assert sum(not r["gold_failed"] for r in rows) == 4960


def test_questions_that_changed(arms):
    f0, f1 = arms
    assert len(f0) == len(f1) == 496
    assert sum(band(f0[q]) != band(f1[q]) for q in f0) == 151
    assert sum(f1[q] > f0[q] for q in f0) == 116
    assert sum(f1[q] < f0[q] for q in f0) == 75


def test_which_questions_moved_table(arms):
    f0, f1 = arms
    moves = Counter((band(f0[q]), band(f1[q]), f0[q] == f1[q]) for q in f0)

    def count(before: str, after: str) -> int:
        return sum(v for (b, a, _), v in moves.items() if (b, a) == (before, after))

    table = {
        "right all ten times, both": count("always", "always"),
        "wrong all ten times, both": count("never", "never"),
        "became right all ten times": count("never", "always") + count("partly", "always"),
        "stopped being right all ten times": count("always", "never") + count("always", "partly"),
        "never right -> partly right": count("never", "partly"),
        "partly right -> never right": count("partly", "never"),
        "partly right both, count moved": moves[("partly", "partly", False)],
        "partly right both, unchanged": moves[("partly", "partly", True)],
    }
    assert list(table.values()) == [67, 230, 49, 27, 46, 29, 40, 8]
    assert sum(table.values()) == 496


def test_flaky_band_is_the_gap(arms):
    # At k = n the gap is, by definition, the share of questions right on 1-9 of 10 attempts.
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    for arm, counts, expected in zip(("baseline", "treatment"), arms, (114, 113), strict=True):
        partly = sum(band(c) == "partly" for c in counts.values())
        assert partly == expected
        assert summary["curves"][arm][-1]["gap"] == pytest.approx(partly / 496)


def test_fine_tuning_headline():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    rounded = {k: [round(v[f] * 100, 1) for f in ("delta", "ci_low", "ci_high")]
               for k, v in summary["differences_at_full_k"].items()}
    assert rounded == {"pass_at_k": [4.2, 0.4, 8.1], "pass_hat_k": [4.4, 1.0, 7.9],
                       "gap": [-0.2, -4.8, 4.4]}
    at_least_once_gained = sum(1 for m in summary["moved_questions"]
                               if m["baseline"] == 0 and m["treatment"] > 0)
    at_least_once_lost = sum(1 for m in summary["moved_questions"]
                             if m["baseline"] > 0 and m["treatment"] == 0)
    assert (at_least_once_gained, at_least_once_lost) == (58, 37)
    assert (summary["moved"]["became_solid"], summary["moved"]["lost_solid"]) == (49, 27)


def test_seven_b_comparison():
    report = json.loads((RESULTS / "failure-modes.json").read_text(encoding="utf-8"))
    always_7b = report["consistency"]["reference"]["always"]
    always_f1 = report["consistency"]["treatment"]["always"]
    assert (always_7b, always_f1) == (179, 116)
    assert (always_7b - always_f1) / always_7b == pytest.approx(0.352, abs=0.001)  # a third

    single = report["differences"]["treatment_minus_reference"]["correct"]
    assert round(single["delta"] * 100, 1) == -8.1
    assert round(single["ci_low"] * 100, 1) == -11.6
    assert round(single["ci_high"] * 100, 1) == -4.4

    crossover = json.loads((RESULTS / "summary-f1-vs-7b.json").read_text(encoding="utf-8"))
    rounded = {k: [round(v[f] * 100, 1) for f in ("delta", "ci_low", "ci_high")]
               for k, v in crossover["differences_at_full_k"].items()}
    assert rounded["pass_at_k"] == [-3.8, -7.9, 0.4]
    assert rounded["pass_hat_k"] == [-12.7, -16.7, -8.5]


def test_trained_parameter_count():
    # LoRA r=16 on q/k/v/o/gate/up/down in each of Qwen2.5-3B's 36 layers:
    # hidden 2048, intermediate 11008, 2 KV heads x 128 = 256 for k and v.
    config = json.loads((RESULTS / "f1-training" / "adapter_config.json").read_text("utf-8"))
    r, layers, hidden, inter, kv = config["r"], 36, 2048, 11008, 256
    shapes = [(hidden, hidden), (hidden, kv), (hidden, kv), (hidden, hidden),
              (hidden, inter), (hidden, inter), (inter, hidden)]
    assert len(config["target_modules"]) == len(shapes)
    assert layers * sum(r * (a + b) for a, b in shapes) == 29_933_568


def test_training_databases():
    manifest = json.loads(
        (RESULTS.parent / "prepared" / "manifest.json").read_text(encoding="utf-8"))
    assert len({e["db_id"] for e in manifest["train_examples"]}) == 65
    assert manifest["kept"] == {"train": 5851, "validation": 367}


def test_published_paths_are_portable():
    for path in RESULTS.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert ":\\\\" not in text and "/root/" not in text, path.name


def test_portable_paths(tmp_path, monkeypatch):
    repo = tmp_path / "other-repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "results").mkdir()
    monkeypatch.chdir(tmp_path)
    assert portable("results/x.jsonl") == "results/x.jsonl"
    assert portable(tmp_path / "a" / "b.json") == "a/b.json"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert portable(repo / "results" / "x.jsonl") == "other-repo/results/x.jsonl"
    assert portable("../other-repo/results/x.jsonl") == "other-repo/results/x.jsonl"
    assert portable("../gguf/model.gguf") == "../gguf/model.gguf"
    assert portable(tmp_path / "blob-sha256") == "blob-sha256"
