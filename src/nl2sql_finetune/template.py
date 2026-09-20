"""The exact text the evaluated model sees, reproduced for training.

This is the most fragile joint in the study. The published baseline (P1) sent each prompt to
Ollama as a single user message with no system message. Ollama's `qwen2.5-coder:3b` Modelfile
then applies its own template -- and that template **injects a default system prompt** that
appears nowhere in P1's code:

    SYSTEM You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

So every prompt the baseline model answered began with that line. A fine-tuned model trained
on anything else would be evaluated on a format it never saw in training, and any difference in
the results could come from the format rather than from the training.

`render` reproduces the rendered string. It is checked two independent ways in the tests:
against the Hugging Face tokenizer's own chat template, and against the prompt-token counts
Ollama logged for every one of P1's attempts.
"""

from __future__ import annotations

# Copied verbatim from `ollama show qwen2.5-coder:3b --modelfile` (Ollama 0.34.1, 2026-09-19).
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def render(prompt: str, *, system: str = SYSTEM) -> str:
    """The prompt as the model receives it, up to the point where it starts writing.

    Ollama's Go template, for one user message and a Modelfile SYSTEM, trims to exactly:
    a system turn, a user turn, and an open assistant turn ending in a newline.
    """
    return (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{prompt}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def completion(sql: str) -> str:
    """The training target: the answer in the format P1's instructions demand, then stop.

    P1's instructions say "Output only the SQL, inside a ```sql code block", and its extractor
    reads that shape first. Ending on `<|im_end|>` teaches the model to stop where Ollama stops
    generating; without it a fine-tuned model can run on past the answer.
    """
    return f"```sql\n{sql.strip()}\n```{IM_END}"
