"""Paths to the things this study reads but does not ship.

Everything large lives outside the repo and is located through environment variables, so the
tests skip cleanly on a machine that hasn't downloaded it rather than failing obscurely.

    P1_DIR        a clone of nl2sql-reliability at the pinned commit, with its data/ fetched
    MODEL_DIR     Qwen2.5-Coder-3B-Instruct from Hugging Face (only the tokenizer is read here)
    TRAIN_JSONL   birdsql/bird23-train-filtered, data/train-00000-of-00001.jsonl
    TRAIN_DBS     the directory BIRD's train.zip databases were extracted into
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _path(variable: str) -> Path | None:
    value = os.environ.get(variable)
    return Path(value) if value and Path(value).exists() else None


@pytest.fixture(scope="session")
def p1_dir() -> Path:
    path = _path("P1_DIR")
    if path is None or not (path / "data" / "arcwise_plat_sql.json").exists():
        pytest.skip("set P1_DIR to a nl2sql-reliability clone with data/ fetched")
    return path


@pytest.fixture(scope="session")
def tokenizer():
    path = _path("MODEL_DIR")
    if path is None:
        pytest.skip("set MODEL_DIR to the Qwen2.5-Coder-3B-Instruct directory")
    from nl2sql_finetune.tokens import Tokenizer

    return Tokenizer(path)


@pytest.fixture(scope="session")
def train_jsonl() -> Path:
    path = _path("TRAIN_JSONL")
    if path is None:
        pytest.skip("set TRAIN_JSONL to the bird23-train-filtered jsonl")
    return path


@pytest.fixture(scope="session")
def train_dbs() -> Path:
    path = _path("TRAIN_DBS")
    if path is None:
        pytest.skip("set TRAIN_DBS to the extracted BIRD train databases")
    return path
