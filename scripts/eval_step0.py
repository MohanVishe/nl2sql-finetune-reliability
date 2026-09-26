"""Step-0 validation loss: the untrained model, measured exactly as training measured it.

    python scripts/eval_step0.py --model-dir ../models/Qwen2.5-Coder-3B-Instruct \
        --data prepared --adapter runs/f1/adapter --out results/f1-training/step0-eval.json

Builds the same SFTTrainer as `train.py` -- same 4-bit NF4 quantisation, same tokenizer and
template, same 3,072-token limit, same completion-only loss, same eval batch size -- and calls
`evaluate()` on the 367-example validation split without training a single step.

Two arms:
- **step 0**: the base model with a freshly initialised LoRA adapter. LoRA initialises its B
  matrices to zero, so the adapter adds exactly nothing; this is the model training started from.
  The script checks that every `lora_B` weight is zero before evaluating.
- **selected adapter** (optional, `--adapter`): the saved step-200 adapter, re-evaluated by the
  same code as a check that this script reproduces the logged 0.1473.

No weights are updated.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def build_trainer(model, tokenizer, dataset, out: Path, batch: int, max_length: int,
                  peft_config=None):
    from trl import SFTConfig, SFTTrainer

    config = SFTConfig(
        output_dir=str(out),
        per_device_eval_batch_size=batch,
        bf16=True,
        max_length=max_length,
        completion_only_loss=True,
        packing=False,
        report_to="none",
        seed=0,
        dataloader_num_workers=0,
    )
    return SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def load_base(model_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("prepared"))
    parser.add_argument("--adapter", type=Path, help="saved adapter to re-evaluate as a check")
    parser.add_argument("--out", type=Path, required=True, help="result JSON")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=3072)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("error: no CUDA device")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.data / "train.jsonl"),
                    "validation": str(args.data / "validation.jsonl")},
    )
    scratch = args.out.parent / "_eval_scratch"
    record: dict = {
        "validation_examples": len(dataset["validation"]),
        "loss": "mean token cross-entropy over completion tokens only (completion_only_loss=True)",
        "quantisation": "bitsandbytes 4-bit NF4, double quant, bf16 compute",
        "max_length": args.max_length,
        "eval_batch_size": args.batch,
    }

    # Step 0: base + zero-initialised LoRA (identical to the base model's function).
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = build_trainer(load_base(args.model_dir), tokenizer, dataset, scratch,
                            args.batch, args.max_length, peft_config)
    lora_b = [p for n, p in trainer.model.named_parameters() if "lora_B" in n]
    if not lora_b or any(p.abs().max().item() != 0 for p in lora_b):
        raise SystemExit("error: fresh LoRA B matrices are not all zero; step 0 would not be the base")
    started = time.perf_counter()
    step0 = trainer.evaluate()
    record["step0"] = {**step0, "seconds": time.perf_counter() - started,
                       "lora_B_tensors_checked_zero": len(lora_b)}
    print("step 0:", step0)
    del trainer
    torch.cuda.empty_cache()

    if args.adapter:
        model = PeftModel.from_pretrained(load_base(args.model_dir), args.adapter)
        trainer = build_trainer(model, tokenizer, dataset, scratch, args.batch, args.max_length)
        started = time.perf_counter()
        sel = trainer.evaluate()
        record["selected_adapter"] = {**sel, "seconds": time.perf_counter() - started,
                                      "adapter": args.adapter.name}
        print("selected adapter:", sel)

    import peft
    import transformers
    import trl
    record["environment"] = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "transformers": transformers.__version__, "trl": trl.__version__,
        "peft": peft.__version__, "platform": platform.platform(),
    }
    args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
