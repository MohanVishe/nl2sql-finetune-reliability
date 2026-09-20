# Run log

What was actually run, on what, and what went wrong. Written so that someone else can judge
the numbers rather than take them on trust. Newest first.

## The machine

| | |
|---|---|
| GPU | NVIDIA RTX 3070, 8 GB |
| Training | WSL2 Ubuntu 24.04, Python 3.12, torch 2.14.0+cu126, transformers 5.17.0, trl 1.13.0, peft 0.21.0, bitsandbytes 0.50.2 |
| Serving and evaluation | Windows 11, Ollama 0.34.2 |
| Conversion | llama.cpp release b11042 (converter at commit `ec92815`) |
| Harness | [`nl2sql-reliability`](https://github.com/MohanVishe/nl2sql-reliability) at `a0ea7fb`, unchanged |

Training runs in WSL2 because Windows Smart App Control (a security feature that blocks
unsigned programs) refuses to load PyTorch's own libraries and llama.cpp's quantiser on this
machine. The security setting was left on; the build moved instead.

## Data

- Training: `birdsql/bird23-train-filtered`, 6,601 examples over 69 databases, plus BIRD's
  `train.zip` (8,919,543,554 bytes, matching the published size, all files CRC-checked).
- 383 examples dropped for length, all from `works_cycles`, whose schema alone is about 5,000
  tokens. Dropped, never truncated.
- Final split: **5,851 training / 367 validation**, the validation set being three whole
  databases (`image_and_language`, `retail_complains`, `video_games`) the model never trains on.
- Contamination check against the evaluation set: **0 shared databases, 0 shared questions.**
- Exact contents: `prepared/manifest.json`.

## Training (F1)

| | |
|---|---|
| Started | 2026-09-19 08:14 IST |
| Finished | 2026-09-19 17:22 IST (3.48 h of compute; the machine slept in between) |
| Steps | 732 (2 epochs), effective batch 16, lr 1e-4 cosine, LoRA r=16 α=32 on all seven projections |
| Peak VRAM | 4.10 GB |
| Best checkpoint | **step 200**, validation loss 0.1473 |
| Final checkpoint | step 732, validation loss 0.1544 |

Validation loss bottomed at step 200 and rose through epoch 2 while training loss kept falling,
so the step-200 checkpoint was kept. The saved adapter was verified to be that checkpoint:
504 tensors, maximum absolute difference 0.

A 20-step smoke run preceded it. Its purpose was the loss mask: the supervised tokens were
decoded and had to be exactly the fenced SQL answer and the end-of-turn marker.

## Building the models

Four differences between this project's pipeline and the published baseline were found and
removed. Each was caught by a check, not by inspection:

1. **Q4_K_M layer mix.** The format is a mixture of 4-bit and 6-bit blocks, and the current
   llama.cpp chooses different layers for the 6-bit blocks than the build Ollama published:
   404 of 434 tensors matched, 30 differed only in precision. Fixed by copying the reference
   file's per-tensor layout (`to_gguf.py --match-layout`).
2. **Sampling defaults.** The converter copies `general.sampling.*` (temperature 0.7, top_p 0.8,
   top_k 20, repeat penalty 1.05) out of `generation_config.json`; the reference file has none.
   A runtime that reads them would sample these models differently from the baseline. Stripped
   before quantising, and `gguf_compare.py` now diffs every metadata key rather than a chosen
   list — the chosen list is how this one slipped through the first time.
3. **CRLF in the chat template.** Windows newline translation put carriage returns into the
   Ollama Modelfile, which would have changed every prompt. The post-build verification refused
   the model until it was written with `\n`.
4. **A dropped tokenizer flag.** `transformers` 5's `save_pretrained` omits
   `add_bos_token: false`, so the merged model's file lacked
   `tokenizer.ggml.add_bos_token` and the runtime's default would decide whether the fine-tuned
   model saw a start token the baseline never saw. `merge.py` now copies tokenizer files
   byte-for-byte.

Result: **F0 is byte-identical to Ollama's published `qwen2.5-coder:3b`** — all 434 tensors and
every metadata key (`results/f0-vs-reference.json`). F1 differs from F0 in exactly the 252
tensors LoRA trained (7 projections × 36 layers); the other 182 are byte-identical.

## Evaluation

Both arms: 498 questions × 10 attempts, temperature 0.2, context 8,192, at most 512 new tokens,
single turn. Two questions are excluded from scoring because their official query times out at
30 s, leaving **496 scored questions** — the same exclusions the published baseline made.

| Arm | Attempts | Result |
|---|---|---|
| F0 (`p3-f0-base`) | 4,980 | Complete. pass@10 41.7%, pass^10 19.0%, gap 22.8% |
| F1 (`p3-f1-qlora`) | 4,980 | Complete. pass@10 46.2%, pass^10 23.4%, gap 22.8% |

F0 against the published baseline (`results/f0-vs-c.txt`): pass@10 −0.6 [−2.6, +1.6],
pass^10 +0.2 [−1.6, +2.2], gap −0.8 [−3.6, +2.0]. All intervals include zero, so neither this
pipeline nor the Ollama update (0.34.1 → 0.34.2 mid-study) moves the numbers.

F1 ran from 2026-09-19 21:49 IST to 2026-09-20 19:28 IST, exit code 0, across four stops (three
bugchecks and one planned restart, below). Both result files were checked after the final
attempt: 4,980 rows each, 4,980 unique `(arm, question, attempt)` keys, no duplicates, no
unparsable lines, and 20 `gold_failed` rows each — the two excluded questions × ten attempts.

F1 against F0 (`results/summary.json`): pass@10 +4.4 [+0.6, +8.3], pass^10 +4.4 [+1.0, +7.9],
gap +0.0 [−4.6, +4.4].

## Incidents

- **Ollama updated itself** from 0.34.1 to 0.34.2 between the published baseline and this study.
  This is why F0 exists and why `evaluate.py` records the server version and refuses to resume
  an arm on a different one.
- **Three Windows bugchecks** during evaluation (2026-09-19 20:07, 22:00, 23:07; codes 0x7E,
  0x133, 0xD1). The machine's event log shows 17 in 60 days with varied codes, 15 of them
  starting before this project — a machine problem, not a study problem. Each stop cost nothing:
  the harness resumes from its completed attempts, and no partial line was ever written. The
  files were checked for duplicate `(question, attempt)` pairs after every resume: none.
- **Planned restart mid-run** (2026-09-20 18:23 IST). The evaluation was stopped deliberately
  at 4,198 of 4,980 attempts so the machine could be rebooted. Order matters: the watchdog was
  stopped first, or it would have relaunched the run within the minute, and only then the worker
  process tree. The file was checked while idle — 4,198 rows, no duplicate keys, last line
  complete. It resumed 30 minutes later at question 1,171 attempt 7, the attempt immediately
  after the last completed one, and ran the remaining 782 attempts at 0.36 attempts/second.
  Ollama had to be restarted after the reboot and came back on the same 0.34.2; had it
  self-updated in the meantime, `evaluate.py` would have refused to resume the arm rather than
  mixing two server versions in one result file.

- **GPU contention.** An unrelated 7B model was served on the same 8 GB card for a few minutes;
  Ollama evicted and reloaded a model per request, and throughput fell from 0.35 to 0.15
  attempts per second. The evaluation was paused and resumed once the card was free. Outputs are
  unaffected — every load put all 37 layers on the GPU — but the two must not share a card.
