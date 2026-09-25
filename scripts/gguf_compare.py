"""Compare two GGUF files tensor by tensor: names, shapes, quantisation types and raw bytes.

    uv run python scripts/gguf_compare.py ../gguf/f0-base.Q4_K_M.gguf --reference qwen2.5-coder:3b

`--reference` accepts a path, or an Ollama model name whose weights blob is looked up through
`ollama show --modelfile`. Used to establish that F0 -- the base model sent through this
project's own conversion pipeline -- is the model P1 measured, before any score is compared.

Reports, in order of severity: tensors missing on either side, shape or type mismatches, then
how many tensors are byte-identical, then every metadata key that differs (arrays such as the
vocabulary and merges are compared by hash). Exits non-zero on any structural or metadata
difference. Key and tensor *order* in the file is not compared: it does not change the model,
and it is why two equivalent files can have different sha256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from nl2sql_finetune.paths import portable

TOOLS = Path("../tools")


def resolve(reference: str) -> Path:
    path = Path(reference)
    if path.exists():
        return path
    modelfile = subprocess.run(["ollama", "show", reference, "--modelfile"], check=True,
                               capture_output=True, text=True, encoding="utf-8").stdout
    for line in modelfile.splitlines():
        if line.startswith("FROM ") and "blobs" in line:
            return Path(line[5:].strip())
    raise SystemExit(f"error: no weights blob in the Modelfile of {reference}")


def field_value(field):
    """A metadata field's value as plain Python; long arrays become a sha256 digest."""
    from gguf import GGUFValueType

    if not field.types:
        return None
    if field.types[0] == GGUFValueType.ARRAY:
        digest = hashlib.sha256()
        for index in field.data:
            digest.update(bytes(field.parts[index]))
        return f"array[{len(field.data)}] sha256:{digest.hexdigest()[:16]}"
    value = field.parts[field.data[0]]
    if field.types[0] == GGUFValueType.STRING:
        return bytes(value).decode("utf-8")
    return value.tolist()[0] if hasattr(value, "tolist") else value


def load(path: Path):
    from gguf import GGUFReader

    reader = GGUFReader(path)
    # Every key, not a chosen list: a hand-picked list is how extra sampling defaults slipped
    # past the first version of this check. GGUF.* entries are container bookkeeping.
    meta = {key: field_value(field) for key, field in reader.fields.items()
            if not key.startswith("GGUF.")}
    tensors = {t.name: t for t in reader.tensors}
    return meta, tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--tools", type=Path, default=TOOLS)
    parser.add_argument("--json", type=Path, help="also write the report here")
    args = parser.parse_args()

    # Same gguf-py as the converter that wrote the candidate.
    sys.path.insert(0, str(args.tools / "llama-src" / "gguf-py"))

    reference_path = resolve(args.reference)
    print(f"candidate: {args.candidate}\nreference: {reference_path}")
    cand_meta, cand = load(args.candidate)
    ref_meta, ref = load(reference_path)

    meta_diff = {k: {"candidate": cand_meta.get(k), "reference": ref_meta.get(k)}
                 for k in sorted(set(cand_meta) | set(ref_meta))
                 if cand_meta.get(k) != ref_meta.get(k)}
    missing = sorted(set(ref) - set(cand))
    extra = sorted(set(cand) - set(ref))
    shape_or_type = []
    identical = 0
    differing_bytes = []
    type_counts: dict[str, int] = {}
    for name in sorted(set(ref) & set(cand)):
        a, b = cand[name], ref[name]
        type_counts[b.tensor_type.name] = type_counts.get(b.tensor_type.name, 0) + 1
        if list(a.shape) != list(b.shape) or a.tensor_type != b.tensor_type:
            shape_or_type.append({"tensor": name,
                                  "candidate": [a.tensor_type.name, list(map(int, a.shape))],
                                  "reference": [b.tensor_type.name, list(map(int, b.shape))]})
        elif bytes(a.data) == bytes(b.data):
            identical += 1
        else:
            differing_bytes.append(name)

    common = len(set(ref) & set(cand))
    report = {
        "candidate": portable(args.candidate), "reference": portable(reference_path),
        "tensors": {"reference": len(ref), "candidate": len(cand), "common": common,
                    "missing_in_candidate": missing, "extra_in_candidate": extra,
                    "shape_or_type_mismatch": shape_or_type,
                    "byte_identical": identical, "bytes_differ": differing_bytes,
                    "reference_types": type_counts},
        "metadata_differences": meta_diff,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"tensors: {len(cand)} candidate / {len(ref)} reference, {common} in common")
    print(f"reference quantisation types: {type_counts}")
    print(f"missing {len(missing)}, extra {len(extra)}, shape/type mismatches {len(shape_or_type)}")
    for row in shape_or_type[:10]:
        print("  ", row)
    print(f"byte-identical {identical}/{common}; bytes differ in {len(differing_bytes)}")
    for name in differing_bytes[:10]:
        print("  ", name)
    print(f"metadata differences: {json.dumps(meta_diff, default=str) if meta_diff else 'none'}")
    structural = missing or extra or shape_or_type or meta_diff
    return 1 if structural else 0


if __name__ == "__main__":
    raise SystemExit(main())
