"""Build the training and validation sets, and prove they are fit to train on.

    uv run python scripts/prepare_data.py --p1-dir ../nl2sql-reliability \
        --train-jsonl ../data/bird23-train-filtered/data/train-00000-of-00001.jsonl \
        --train-dbs ../data/train --model-dir ../models/Qwen2.5-Coder-3B-Instruct

Refuses to write anything if the training data overlaps the evaluation set. Drops -- never
truncates -- examples longer than --max-tokens, because a truncated example teaches the model
an answer to a question it was never fully shown. Every decision is written to
`prepared/manifest.json` so the exact training set is checkable by anyone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from nl2sql_finetune import data

# The Hugging Face revision of birdsql/bird23-train-filtered this study trained on. The file's
# sha256 is recorded next to it, so a copy downloaded later can be checked against it.
DATASET = "birdsql/bird23-train-filtered"
DATASET_REVISION = "4068469807b255fcfc0816bdd520946fe460d256"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p1-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, default=data.DEFAULT_TRAIN)
    parser.add_argument("--train-dbs", type=Path, default=data.DEFAULT_TRAIN_DATABASES)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("prepared"))
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="pilot: keep only N train examples")
    parser.add_argument("--source-revision", default=DATASET_REVISION,
                        help=f"the {DATASET} revision --train-jsonl was downloaded at")
    args = parser.parse_args()

    from nl2sql_reliability import dataset

    from nl2sql_finetune.tokens import Tokenizer

    tokenizer = Tokenizer(args.model_dir)

    examples = data.load_train(args.train_jsonl)
    evaluation = dataset.load(args.p1_dir / "data" / "arcwise_plat_sql.json")
    report = data.contamination(examples, evaluation)
    print(f"loaded {len(examples)} training examples over "
          f"{len({e.db_id for e in examples})} databases")
    print(f"contamination: {len(report.shared_databases)} shared databases, "
          f"{len(report.shared_questions)} shared questions")
    data.assert_clean(examples, evaluation)

    # Length filtering happens *before* the split. Filtering after it let one huge-schema
    # database (works_cycles, ~5k tokens per prompt) be drawn for validation and then dropped
    # wholesale, leaving validation a third of its intended size.
    records: dict[int, dict] = {}
    dropped: list[dict] = []
    lengths: list[int] = []
    boundary_shifts = 0
    for index, example in enumerate(examples):
        record = data.pair(example, args.train_dbs)
        prompt_ids = tokenizer.encode(record["prompt"])
        full_ids = tokenizer.encode(record["prompt"] + record["completion"])
        # The loss mask assumes the prompt's tokens are a prefix of the full sequence's.
        if full_ids[: len(prompt_ids)] != prompt_ids:
            boundary_shifts += 1
        length = len(full_ids)
        lengths.append(length)
        ident = {"index": index, "db_id": example.db_id,
                 "question_sha1": hashlib.sha1(example.question.encode()).hexdigest()[:12],
                 "tokens": length}
        if length > args.max_tokens:
            dropped.append(ident)
            continue
        records[id(example)] = {**record, **ident}

    usable = [e for e in examples if id(e) in records]
    train, validation = data.split_by_database(usable, args.holdout, seed=args.seed)
    print(f"dropped {len(dropped)} over {args.max_tokens} tokens "
          f"(databases: {sorted({d['db_id'] for d in dropped})})")
    print(f"split by database: {len(train)} train / {len(validation)} validation "
          f"({len({e.db_id for e in validation})} held-out databases)")
    kept = {"train": [records[id(e)] for e in train],
            "validation": [records[id(e)] for e in validation]}

    if boundary_shifts:
        print(f"error: {boundary_shifts} examples tokenise differently at the prompt/completion "
              "boundary; the loss mask would be wrong.", file=sys.stderr)
        return 1

    if args.limit:
        kept["train"] = kept["train"][: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in kept.items():
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps({"prompt": row["prompt"], "completion": row["completion"]},
                                    ensure_ascii=False) + "\n")

    quantiles = statistics.quantiles(lengths, n=100)
    manifest = {
        "source": str(args.train_jsonl.name),
        "source_dataset": DATASET,
        "source_revision": args.source_revision,
        "source_sha256": hashlib.sha256(args.train_jsonl.read_bytes()).hexdigest(),
        "examples_loaded": len(examples),
        "contamination": {"shared_databases": sorted(report.shared_databases),
                          "shared_questions": len(report.shared_questions)},
        "holdout_fraction": args.holdout,
        "seed": args.seed,
        "validation_databases": sorted({e.db_id for e in validation}),
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "token_lengths": {"min": min(lengths), "median": statistics.median(lengths),
                          "p95": quantiles[94], "p99": quantiles[98], "max": max(lengths)},
        "kept": {split: len(rows) for split, rows in kept.items()},
        "dropped_over_length": len(dropped),
        "dropped": dropped,
        "train_examples": [{k: r[k] for k in ("index", "db_id", "question_sha1", "tokens")}
                           for r in kept["train"]],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"token lengths: median {manifest['token_lengths']['median']:.0f}, "
          f"p95 {quantiles[94]:.0f}, p99 {quantiles[98]:.0f}, max {max(lengths)}")
    print(f"wrote {len(kept['train'])} train / {len(kept['validation'])} validation to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
