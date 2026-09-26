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
| Harness | [`nl2sql-reliability`](https://github.com/MohanVishe/nl2sql-reliability): generated at `a0ea7fb`, re-scored at `ed25506` (scorer fixes only, see below), used unchanged |

Training runs in WSL2 because Windows Smart App Control (a security feature that blocks
unsigned programs) refuses to load PyTorch's own libraries and llama.cpp's quantiser on this
machine. The security setting was left on; the build moved instead.

## Data

- Training: `birdsql/bird23-train-filtered` at revision `4068469807b2` (file sha256 in the
  manifest), 6,601 examples over 69 databases, plus BIRD's
  `train.zip` (8,919,543,554 bytes, matching the published size, all files CRC-checked).
- 383 examples dropped for length, all from `works_cycles`, whose schema alone is about 5,000
  tokens. Dropped, never truncated.
- Final split: **5,851 training examples over 65 databases / 367 validation**, the validation
  set being three whole databases (`image_and_language`, `retail_complains`, `video_games`) the
  model never trains on.
- Evaluation set: Arcwise-Plat-SQL (498 BIRD Mini-Dev questions with a corrected SQL answer
  key, CC BY-SA 4.0), read from the pinned harness clone.
- Contamination check against the evaluation set: **0 shared databases, 0 shared questions.**
- Exact contents: `prepared/manifest.json`.

## Training (F1)

| | |
|---|---|
| Started | 2026-09-19 08:14 IST |
| Finished | 2026-09-19 17:22 IST (3.48 h of compute; the machine slept in between) |
| Steps | 732 (2 epochs, all completed), effective batch 16, lr 1e-4 cosine, LoRA r=16 α=32 on all seven projections (29.9M trained parameters) |
| Peak VRAM | 4.10 GB |
| Selected checkpoint | **step 200**, validation loss 0.1473 — the lowest of the 15 evaluations |
| Final checkpoint | step 732, validation loss 0.1544 |

The run trained all 732 steps. Validation loss was evaluated every 50 steps; it was lowest at
step 200 (steps 200–350 were all within 0.002 of it) and rose through epoch 2 while training
loss kept falling. `load_best_model_at_end` then selected the step-200 checkpoint on held-out
validation loss — checkpoint selection, not early stopping. The saved adapter was verified to be
that checkpoint: 504 tensors, maximum absolute difference 0. Training logged no step-0
evaluation; it was measured afterwards (see "Step-0 validation loss" below).

A 20-step smoke run preceded it. Its purpose was the loss mask: the supervised tokens were
decoded and had to be exactly the fenced SQL answer and the end-of-turn marker.

## Step-0 validation loss (2026-09-26)

Measured after training, without retraining: `scripts/eval_step0.py` builds the same
`SFTTrainer` as `train.py` (bitsandbytes 4-bit NF4 with double quantisation, bf16 compute,
3,072-token limit, completion-only loss, eval batch 2) and calls `evaluate()` on the 367
validation examples. Step 0 is the base model with a freshly initialised LoRA adapter; the
script checks that all 252 `lora_B` matrices are zero, so the adapter adds nothing. As a check,
it then re-evaluates the saved step-200 adapter. Same WSL2 environment as training, RTX 3070,
about 65 s per evaluation. Result: `results/f1-training/step0-eval.json`.

| | Validation loss | Answer tokens predicted exactly |
|---|---:|---:|
| Step 0 (untrained) | **0.2385** | 93.1% |
| Step 50 (logged in training) | 0.1544 | — |
| Step 200, selected (logged in training) | 0.1473 | — |
| Step 200, re-evaluated by this script | 0.14731 (identical to the logged 0.147310) | 95.6% |

Training lowered held-out loss by 0.0912, 38% of the starting value; 92% of that drop had
happened by step 50. The evaluation is deterministic on a fixed split, so it has no sampling
interval; the three validation databases are one draw, and a different held-out set would give
different values.

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

Result: **all 434 of F0's tensors are byte-identical to Ollama's published `qwen2.5-coder:3b`,
and every metadata key has the same value** (`results/f0-vs-reference.json`). The files
themselves are not byte-identical: the metadata keys are written in a different order, so the
SHA-256 of F0 (`916368a3…`) differs from the reference blob's (`4a188102…`) at the same size,
1,929,903,072 bytes. F1 differs from F0 in exactly the 252
tensors LoRA trained (7 projections × 36 layers); the other 182 are byte-identical.

## Evaluation

Both arms: 498 questions × 10 attempts, temperature 0.2, context 8,192, at most 512 new tokens,
single turn. Two questions are excluded from scoring because their reference query times out at
30 s, leaving **496 scored questions** — the same exclusions the published baseline made.

| Arm | Attempts | Result |
|---|---|---|
| F0 (`p3-f0-base`) | 4,980 (4,960 scored) | Complete. pass@10 41.9%, pass^10 19.0%, gap 23.0% |
| F1 (`p3-f1-qlora`) | 4,980 (4,960 scored) | Complete. pass@10 46.2%, pass^10 23.4%, gap 22.8% |

F0 against the published baseline (`results/summary-f0-vs-c.json`): pass@10 −0.6 [−2.8, +1.6],
pass^10 +0.2 [−1.6, +2.0], gap −0.8 [−3.6, +2.0]. The earlier project's own `compare_arms.py`
(`results/f0-vs-c.txt`, a different resampling seed) agrees: [−2.6, +1.6], [−1.6, +2.2],
[−3.6, +2.0]. All intervals include zero, so neither this
pipeline nor the Ollama update (0.34.1 → 0.34.2, between the earlier study and this one; both
arms here ran on 0.34.2) moves the numbers.

F0 ran from 2026-09-19 17:31 to 21:49 IST, with three stops: a GPU-contention pause
(17:34–17:40), a pause at the owner's request (18:00–18:15) and a bugcheck (20:05–20:10).
F1 ran from 2026-09-19 21:49 IST to 2026-09-20 19:28 IST, exit code 0, with four stops: bugchecks
at 22:00 (resumed 22:16), 23:07 (resumed the next day at 16:07) and 16:59 (resumed 17:09), and
the planned restart at 18:23 (resumed 18:53). Stop and resume points are read from the per-row
timestamps in the result files and the Windows event log. Both result files were checked after
the final attempt: 4,980 rows each, 4,980 unique `(arm, question, attempt)` keys, no duplicates, no
unparsable lines, and 20 `gold_failed` rows each — the two excluded questions × ten attempts.

F1 against F0 (`results/summary.json`): pass@10 +4.2 [+0.4, +8.1], pass^10 +4.4 [+1.0, +7.9],
gap −0.2 [−4.8, +4.4].

Attempts were also split by how they failed (`results/failure-modes.json`, seed 0, same paired
bootstrap): crashes 38.0% → 18.2% [−23.2, −16.5], silent wrong answers 32.3% → 46.9%
[+11.2, +18.0], right 29.7% → 34.9% [+2.2, +8.3]. Questions by consistency: never/flaky/always
is 288/114/94 for F0, 267/113/116 for F1, 248/69/179 for the prompted 7B.

F1 against the published prompted 7B (`results/summary-f1-vs-7b.json`), P1 arm A, read from
`local-7b-single.jsonl` and never re-run: pass@10 −3.8 [−7.9, +0.4], pass^10 −12.7
[−16.7, −8.5], gap +8.9 [+4.2, +13.5]. Single-attempt accuracy (the `correct` rate,
`results/failure-modes.json`): −8.1 [−11.6, −4.4]. This comparison spans Ollama 0.34.1 (the 7B)
and 0.34.2 (F1); F0 is the control that licenses it.

## Re-scoring (2026-09-26)

The earlier project fixed two scorer bugs (its commit `66a8dfd`, pinned here at `ed25506`):
row order is now enforced only when the reference query has a **top-level** `ORDER BY` (one in
a subquery or window used to force it too), and empty or comment-only SQL counts as an
execution error instead of an empty result. Both arms were re-scored from their recorded SQL
with its `scripts/rescore.py`, which re-executes every attempt and every reference query and
refuses to write if any attempt's execution status would change. None did; nothing was
generated again.

| Arm | Verdicts changed |
|---|---|
| F0 | 1 — question 728, attempt 3: wrong → right (`exact column order matches`) |
| F1 | 0 |

The earlier project's published 7B and 3B attempts were re-scored the same way on its side
(17 verdicts, all on question 728). Every report and chart here was regenerated from the
re-scored files. What moved: F0 pass@10 41.7 → 41.9 and gap 22.8 → 23.0; F1 − F0 pass@10
+4.4 → +4.2 and gap +0.0 → −0.2 [−4.8, +4.4]; F0's flaky band 113 → 114; the 7B pass@10
49.8 → 50.0 and pass@1 42.9 → 43.0, so F1 − 7B pass@10 −3.6 → −3.8 [−7.9, +0.4] and pass@1
−8.0 → −8.1 [−11.6, −4.4]. pass^10 did not change for any arm. Every conclusion stands: the gap
change still includes zero, the capability rise still excludes it, and pass@10 against the 7B
still cannot be separated. `results/<arm>.run.json` records the re-score.

## Incidents

- **Ollama updated itself** from 0.34.1 to 0.34.2 between the published baseline and this study.
  This is why F0 exists and why `evaluate.py` records the server version and refuses to resume
  an arm on a different one.
- **Four Windows bugchecks** during evaluation: 2026-09-19 20:05 (0x7E, during F0), 22:00
  (0x133) and 23:07 (0xD1), and 2026-09-20 16:59 (0xF7), the last three during F1. The
  machine's event log shows 20 bugchecks from 2026-07-20 to the end of the evaluation, with
  varied codes, 15 of them before this project began — a machine problem, not a study problem.
  Each stop cost nothing: the harness resumes from its completed attempts, and no partial line
  was ever written. The
  files were checked for duplicate `(question, attempt)` pairs after every resume: none.
- **Planned restart mid-run** (2026-09-20 18:23 IST). The evaluation was stopped deliberately
  at 4,198 of 4,980 attempts so the machine could be rebooted. Order matters: the watchdog was
  stopped first, or it would have relaunched the run within the minute, and only then the worker
  process tree. The file was checked while idle — 4,198 rows, no duplicate keys, last line
  complete. It resumed 30 minutes later with row 4,199 — question 1169, attempt index 8 (its
  ninth attempt), the attempt immediately after the last completed one — and ran the remaining
  782 attempts at 0.38 attempts/second.
  Ollama had to be restarted after the reboot and came back on the same 0.34.2; had it
  self-updated in the meantime, `evaluate.py` would have refused to resume the arm rather than
  mixing two server versions in one result file.

- **GPU contention** (F0, 2026-09-19 17:30–17:40). An unrelated 7B model was served on the same
  8 GB card for a few minutes; Ollama evicted and reloaded a model per request, and throughput
  fell from 0.35 to 0.15 attempts per second. The evaluation was paused after 27 attempts and
  resumed once the card was free. Outputs are
  unaffected — every load put all 37 layers on the GPU — but the two must not share a card.
