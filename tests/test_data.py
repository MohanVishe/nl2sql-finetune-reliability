from __future__ import annotations

from dataclasses import dataclass

import pytest

from nl2sql_finetune.data import (
    ContaminationError,
    Example,
    assert_clean,
    contamination,
    load_train,
    normalise,
    split_by_database,
)


@dataclass
class _Question:  # the two fields of P1's Question that contamination reads
    db_id: str
    question: str


def _examples(spec: dict[str, int]) -> list[Example]:
    return [
        Example(db_id=db, question=f"{db} question {i}", evidence="", sql="SELECT 1")
        for db, n in spec.items()
        for i in range(n)
    ]


def test_normalise_ignores_case_punctuation_and_spacing():
    assert normalise("  How MANY schools?  ") == normalise("how many   schools")


def test_contamination_catches_a_shared_database():
    report = contamination(_examples({"a": 2}), [_Question("a", "something else")])
    assert report.shared_databases == {"a"} and not report.clean


def test_contamination_catches_a_reworded_duplicate_question():
    train = [Example("x", "How many rows?", "", "SELECT 1")]
    report = contamination(train, [_Question("y", "how many ROWS")])
    assert report.shared_questions and not report.clean


def test_assert_clean_raises_on_overlap_and_passes_when_disjoint():
    with pytest.raises(ContaminationError):
        assert_clean(_examples({"a": 1}), [_Question("a", "q")])
    assert_clean(_examples({"a": 1}), [_Question("b", "q")])


def test_split_holds_out_whole_databases():
    examples = _examples({f"db{i}": 10 for i in range(20)})
    train, val = split_by_database(examples, holdout=0.1, seed=0)
    assert {e.db_id for e in train}.isdisjoint({e.db_id for e in val})
    assert len(train) + len(val) == len(examples)
    assert 0 < len(val) <= 0.2 * len(examples)


def test_split_is_deterministic_for_a_seed():
    examples = _examples({f"db{i}": 5 for i in range(30)})
    assert split_by_database(examples, seed=7) == split_by_database(examples, seed=7)


@pytest.mark.databases
def test_real_training_set_does_not_touch_the_evaluation_set(train_jsonl, p1_dir):
    from nl2sql_reliability import dataset

    train = load_train(train_jsonl)
    evaluation = dataset.load(p1_dir / "data" / "arcwise_plat_sql.json")
    assert len(train) == 6601
    report = contamination(train, evaluation)
    assert report.clean, (
        f"shared databases: {sorted(report.shared_databases)}; "
        f"shared questions: {len(report.shared_questions)}"
    )
