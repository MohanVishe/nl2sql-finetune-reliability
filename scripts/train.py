"""QLoRA fine-tuning of Qwen2.5-Coder-3B-Instruct on the prepared BIRD training set.

    uv run python scripts/train.py --model-dir ../models/Qwen2.5-Coder-3B-Instruct \
        --data prepared --out runs/f1 --epochs 2

Plain Hugging Face stack -- transformers + peft + trl + bitsandbytes -- so every moving part is
visible. Loss is computed on the completion only: the model is taught to write the SQL, not to
reproduce the schema it was given.

Before training starts, one collated batch is decoded and checked: the tokens that carry loss
must be exactly the answer and its end-of-turn marker. A silently wrong loss mask trains a model
that looks fine and measures nothing, so this refuses to proceed on a mismatch.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("prepared"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N steps")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in --out")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("error: no CUDA device -- this would train on CPU for days")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.data / "train.jsonl"),
                    "validation": str(args.data / "validation.jsonl")},
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # transformers 5: a float < 1 is a ratio of total steps
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=True,
        bf16=True,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,  # Windows: worker processes re-import torch and are slow
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    check_loss_mask(trainer, tokenizer)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    resume = args.resume and any(args.out.glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    elapsed = time.perf_counter() - started

    best = trainer.state.best_model_checkpoint
    trainer.save_model(str(args.out / "adapter"))
    tokenizer.save_pretrained(str(args.out / "adapter"))

    record = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "train_loss": result.training_loss,
        "best_checkpoint": best,
        "best_eval_loss": trainer.state.best_metric,
        "steps": trainer.state.global_step,
        "wall_clock_hours": elapsed / 3600,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "log_history": trainer.state.log_history,
    }
    (args.out / "training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"done in {elapsed / 3600:.2f} h; best eval loss {trainer.state.best_metric} at {best}")
    print(f"peak VRAM {record['peak_vram_gb']:.2f} GB; adapter saved to {args.out / 'adapter'}")
    return 0


def check_loss_mask(trainer, tokenizer) -> None:
    """Decode the tokens that carry loss in the first training example, and insist they are
    exactly the answer: a ```sql fence, the query, the closing fence, and <|im_end|>."""
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    supervised = tokenizer.decode(labels[labels != -100])
    print("supervised tokens of example 0:", repr(supervised))
    if not supervised.startswith("```sql\n"):
        raise SystemExit(f"error: loss starts somewhere other than the answer: {supervised[:80]!r}")
    if not supervised.rstrip().endswith("```<|im_end|>"):
        raise SystemExit(f"error: loss does not end on the end-of-turn marker: "
                         f"{supervised[-80:]!r}")
    if "<|im_start|>" in supervised or "Schema:" in supervised:
        raise SystemExit("error: the prompt is carrying loss")


if __name__ == "__main__":
    raise SystemExit(main())
