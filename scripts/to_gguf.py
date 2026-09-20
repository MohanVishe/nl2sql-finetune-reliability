"""Convert a Hugging Face checkpoint to a Q4_K_M GGUF with llama.cpp.

    uv run python scripts/to_gguf.py --source ../models/Qwen2.5-Coder-3B-Instruct \
        --out ../gguf/f0-base.Q4_K_M.gguf

Why llama.cpp and not `ollama create --quantize`: Ollama's built-in quantiser only offers int4,
int8, nvfp4, mxfp4 and mxfp8. The baseline P1 measured is Q4_K_M -- a llama.cpp k-quant -- so
producing any other format would change the model under test, not just the pipeline.

Two steps, both from one pinned llama.cpp release (see TOOLS below):
  1. convert_hf_to_gguf.py   safetensors -> GGUF at --outtype (f16 by default)
  2. llama-quantize          GGUF -> Q4_K_M

`--match-layout REF.gguf` pins every tensor's quantisation type to the reference file's. Q4_K_M
is a *mix*: some attn_v / ffn_down layers get Q6_K, chosen by a heuristic that changed between
llama.cpp versions. Without pinning, this build and Ollama's published qwen2.5-coder:3b agree
byte-for-byte on 404 of 434 tensors and differ only in which 15 layers got the extra bits.

The `general.sampling.*` keys the converter copies from generation_config.json (temp 0.7,
top_p 0.8, top_k 20, repeat penalty 1.05) are removed before quantising: the reference GGUF has
none, and a runtime that reads them as defaults would sample F0/F1 differently from the
baseline. After removal, only the tensors can differ.

No importance matrix is used: an imatrix is computed from calibration text, and a text-to-SQL
calibration set would give the fine-tuned model a quantisation advantage the base did not get.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Pinned: binaries from release b11042 (llama-b11042-bin-ubuntu-x64.tar.gz on Linux,
# llama-b11042-bin-win-cpu-x64.zip on Windows), converter from the same tag, commit
# ec928150501c2572fec05cb949061672bb424914. On Windows with Smart App Control enforcing, the
# unsigned binaries are refused, so the build runs under WSL2.
TOOLS = Path("../tools")


def run(command: list[str], env: dict | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="HF safetensors directory")
    parser.add_argument("--out", type=Path, required=True, help="Q4_K_M .gguf to write")
    parser.add_argument("--outtype", default="f16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--tools", type=Path, default=TOOLS, help="holds llama-src/")
    parser.add_argument("--llama-bin", type=Path,
                        default=Path(os.environ.get("LLAMA_BIN", TOOLS / "llama-bin")),
                        help="directory with llama-quantize (env LLAMA_BIN)")
    parser.add_argument("--match-layout", type=Path,
                        help="reference GGUF whose per-tensor quantisation types are copied")
    parser.add_argument("--keep-intermediate", action="store_true")
    args = parser.parse_args()

    converter = args.tools / "llama-src" / "convert_hf_to_gguf.py"
    quantize = args.llama_bin / ("llama-quantize.exe" if os.name == "nt" else "llama-quantize")
    for needed in (converter, quantize):
        if not needed.exists():
            raise SystemExit(f"error: {needed} not found -- see TOOLS in this script")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    intermediate = args.out.with_name(args.out.stem + f".{args.outtype}.gguf")

    # The converter uses the gguf-py next to it (not any pip-installed gguf), which keeps the
    # converter and the library it writes with at the same commit.
    env = {**os.environ, "PYTHONUTF8": "1"}
    env.pop("NO_LOCAL_GGUF", None)
    run([sys.executable, str(converter), str(args.source),
         "--outtype", args.outtype, "--outfile", str(intermediate)], env=env)
    stripped = strip_sampling(intermediate, args.tools, env)

    overrides = []
    if args.match_layout:
        layout = args.out.with_name(args.out.stem + ".tensor-types.txt")
        layout.write_text(tensor_types(args.match_layout, args.tools), encoding="utf-8",
                          newline="\n")
        overrides = ["--tensor-type-file", str(layout)]
    libraries = {"LD_LIBRARY_PATH": f"{args.llama_bin}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    run([str(quantize), *overrides, str(stripped), str(args.out), args.quant],
        env={**os.environ, **libraries})

    if not args.keep_intermediate:
        intermediate.unlink()
        if stripped != intermediate:
            stripped.unlink()
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    return 0


def strip_sampling(gguf_path: Path, tools: Path, env: dict) -> Path:
    """Remove every general.sampling.* key; returns the file to quantise."""
    sys.path.insert(0, str(tools / "llama-src" / "gguf-py"))
    from gguf import GGUFReader

    keys = [k for k in GGUFReader(gguf_path).fields if k.startswith("general.sampling.")]
    if not keys:
        return gguf_path
    out = gguf_path.with_name(gguf_path.stem + ".nosampling.gguf")
    removals = [arg for key in keys for arg in ("--remove-metadata", key)]
    run([sys.executable, str(tools / "llama-src" / "gguf-py" / "gguf" / "scripts" /
                             "gguf_new_metadata.py"),
         *removals, "--force", str(gguf_path), str(out)], env=env)
    return out


def tensor_types(reference: Path, tools: Path) -> str:
    """One anchored `name=type` line per quantised tensor of the reference GGUF."""
    import re

    sys.path.insert(0, str(tools / "llama-src" / "gguf-py"))
    from gguf import GGMLQuantizationType, GGUFReader

    lines = []
    for tensor in GGUFReader(reference).tensors:
        if tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            continue
        lines.append(f"^{re.escape(tensor.name)}$={tensor.tensor_type.name.lower()}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
