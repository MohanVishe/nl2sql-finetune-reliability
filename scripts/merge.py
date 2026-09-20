"""Fold a trained LoRA adapter into the full-precision base model.

    uv run python scripts/merge.py --base ../models/Qwen2.5-Coder-3B-Instruct \
        --adapter runs/f1/adapter --out merged/f1

The adapter was trained against a 4-bit base, but it is merged into the **bf16** base. That is
the standard QLoRA practice, and it is what makes F1 comparable to F0: both then start from
full-precision safetensors and go through the identical `to_gguf.py` (llama.cpp Q4_K_M) and
`build_ollama_model.py` path. Any quantisation effect is therefore shared by both arms.

Runs on the GPU by default (a 3B model in bf16 is ~6 GB; WSL2 here has only 7 GB of RAM), so
run it after training has released the card. `--device cpu` works where RAM allows.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                device_map=args.device)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)

    # Tokenizer files are copied byte-for-byte, never re-saved: transformers 5's save_pretrained
    # drops `add_bos_token: false` from tokenizer_config.json, the GGUF then lacks
    # tokenizer.ggml.add_bos_token, and the runtime falls back to its own default -- so F1
    # could be fed a BOS token F0 never saw. The README carries the licence metadata the
    # converter writes into the GGUF; LICENSE is required on any distributed derivative.
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "README.md", "LICENSE", "generation_config.json"):
        if (args.base / name).exists():
            shutil.copy2(args.base / name, args.out / name)

    print(f"merged {args.adapter} into {args.base} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
