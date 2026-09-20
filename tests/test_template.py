"""The rendered prompt must be byte-identical to what the baseline model saw.

Two independent witnesses, because a single check of a hand-written template is only as good as
the hand that wrote it:

- the model's own chat template (from tokenizer_config.json), and
- the prompt-token count Ollama reported for every attempt in P1's published results.
"""

from __future__ import annotations

import json
from collections import defaultdict

import pytest

from nl2sql_finetune import template


def test_render_has_system_user_and_open_assistant_turn():
    text = template.render("PROMPT")
    assert text == (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nPROMPT<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_completion_is_fenced_sql_ending_the_turn():
    assert template.completion("  SELECT 1;\n") == "```sql\nSELECT 1;\n```<|im_end|>"


def test_completion_is_read_back_by_p1_extractor():
    """If P1's extractor can't read the training target, the model is taught the wrong shape."""
    from nl2sql_reliability.prompt import extract_sql

    sql = "SELECT name FROM t WHERE id = 3"
    reply = template.completion(sql).removesuffix(template.IM_END)
    assert extract_sql(reply) == sql


def test_render_matches_hugging_face_chat_template(tokenizer):
    for content in ["short", "multi\nline\n\nprompt with ```fences```", "Question: x\nSQL:"]:
        expected = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True
        )
        assert template.render(content) == expected


@pytest.mark.databases
def test_render_reproduces_every_prompt_token_count_p1_logged(p1_dir, tokenizer):
    """The strongest check available: Ollama counted the tokens it actually fed the model.

    For every question P1 asked the 3B, rebuild the prompt with P1's own code, render it with
    this module, tokenise it, and compare with Ollama's count. Any mismatch means the training
    format is not the evaluation format.
    """
    from nl2sql_reliability import dataset
    from nl2sql_reliability.prompt import build_for

    logged: dict[str, set[int]] = defaultdict(set)
    with open(p1_dir / "results" / "final" / "local-3b-single.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            # A question whose gold query failed was never sent to the model; its 0 is a
            # placeholder, not a count. P1 excludes these two from scoring as well.
            if row.get("gold_failed"):
                continue
            if row.get("prompt_tokens") is not None:
                logged[str(row["question_id"])].add(int(row["prompt_tokens"]))

    # Ollama's count must itself be stable across a question's ten attempts.
    unstable = {qid: counts for qid, counts in logged.items() if len(counts) != 1}
    assert not unstable, f"Ollama logged varying prompt sizes: {list(unstable.items())[:3]}"

    questions = dataset.load(p1_dir / "data" / "arcwise_plat_sql.json")
    mismatches = []
    for question in questions:
        counts = logged.get(str(question.question_id))
        if not counts:
            continue
        text = template.render(build_for(question, root=p1_dir / "data").text)
        ours = tokenizer.count(text)
        (theirs,) = counts
        if ours != theirs:
            mismatches.append((question.question_id, ours, theirs))

    assert len(logged) == 496, f"{len(logged)} scored questions in P1's results, expected 496"
    assert not mismatches, f"{len(mismatches)} mismatches, first: {mismatches[:5]}"
