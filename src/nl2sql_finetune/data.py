"""Training data: load it, prove it is clean, and turn it into prompt/completion pairs.

Three rules, each enforced in code rather than assumed:

1. **No contamination.** BIRD's train and dev splits use different databases by design, but a
   claim that fine-tuning improved results is worthless if any evaluation question leaked into
   training. `contamination` checks both database ids and normalised question text, and
   `assert_clean` refuses to continue if either overlaps.
2. **Same prompt as evaluation.** Every training prompt is built by P1's own `prompt.build`
   over a schema rendered by P1's own `db.schema_for` -- imported, never copied. A training
   format that differs from the evaluation format by one line invalidates the comparison.
3. **Hold out by database, not by row.** The evaluation asks about 11 databases the model never
   trained on. A validation split that shares databases with training would measure something
   easier, so whole databases are held out.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nl2sql_reliability import prompt as p1_prompt
from nl2sql_reliability.db import schema_for

from . import template

DEFAULT_TRAIN = Path("../data/bird23-train-filtered/data/train-00000-of-00001.jsonl")
DEFAULT_TRAIN_DATABASES = Path("../data/train")


@dataclass(frozen=True)
class Example:
    db_id: str
    question: str
    evidence: str
    sql: str


def load_train(path: Path | str = DEFAULT_TRAIN) -> list[Example]:
    """Read the filtered BIRD training set."""
    examples = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                Example(
                    db_id=row["db_id"],
                    question=row["question"],
                    evidence=row.get("evidence") or "",
                    sql=row["SQL"],
                )
            )
    return examples


_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalise(question: str) -> str:
    """Question text reduced to lowercase words, so trivial edits don't hide a duplicate."""
    return _NON_WORD.sub(" ", question.lower()).strip()


@dataclass(frozen=True)
class Contamination:
    shared_databases: frozenset[str]
    shared_questions: frozenset[str]

    @property
    def clean(self) -> bool:
        return not self.shared_databases and not self.shared_questions


def contamination(train: list[Example], evaluation: list) -> Contamination:
    """Overlap between training examples and evaluation questions.

    `evaluation` is P1's list of `Question`s; only `.db_id` and `.question` are read.
    """
    return Contamination(
        shared_databases=frozenset({e.db_id for e in train} & {q.db_id for q in evaluation}),
        shared_questions=frozenset(
            {normalise(e.question) for e in train} & {normalise(q.question) for q in evaluation}
        ),
    )


class ContaminationError(RuntimeError):
    """Training data overlaps the evaluation set."""


def assert_clean(train: list[Example], evaluation: list) -> None:
    report = contamination(train, evaluation)
    if not report.clean:
        raise ContaminationError(
            f"{len(report.shared_databases)} shared databases "
            f"({', '.join(sorted(report.shared_databases)) or 'none'}), "
            f"{len(report.shared_questions)} shared questions"
        )


def split_by_database(
    examples: list[Example], holdout: float = 0.05, *, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Hold out whole databases until they cover about `holdout` of the examples.

    Deterministic for a given seed. Returns (train, validation).
    """
    counts = Counter(e.db_id for e in examples)
    order = sorted(counts)
    random.Random(seed).shuffle(order)

    target = holdout * len(examples)
    held: set[str] = set()
    covered = 0
    for db_id in order:
        if covered >= target:
            break
        held.add(db_id)
        covered += counts[db_id]

    train = [e for e in examples if e.db_id not in held]
    validation = [e for e in examples if e.db_id in held]
    return train, validation


def prompt_text(example: Example, root: Path | str = DEFAULT_TRAIN_DATABASES) -> str:
    """The user message, built exactly as P1 builds it at evaluation time."""
    schema = schema_for(example.db_id, root)
    return p1_prompt.build(schema, example.question, example.evidence, db_id=example.db_id).text


def pair(example: Example, root: Path | str = DEFAULT_TRAIN_DATABASES) -> dict[str, str]:
    """A prompt/completion record in the rendered form the model will be trained on."""
    return {
        "prompt": template.render(prompt_text(example, root)),
        "completion": template.completion(example.sql),
    }
