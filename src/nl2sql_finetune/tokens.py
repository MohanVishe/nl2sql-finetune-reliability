"""Tokenisation without torch.

The model directory's `tokenizer.json` is loaded with the Rust `tokenizers` library -- the same
engine `transformers`' fast tokenizer wraps -- and its chat template is rendered with Jinja the
way `apply_chat_template` does. Data preparation and the format tests then need neither torch
nor a GPU, so they run anywhere, including where native torch cannot load.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path


class Tokenizer:
    def __init__(self, model_dir: Path):
        from tokenizers import Tokenizer as _Rust

        self.model_dir = Path(model_dir)
        self._rust = _Rust.from_file(str(self.model_dir / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        return self._rust.encode(text, add_special_tokens=False).ids

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def decode(self, ids: list[int]) -> str:
        return self._rust.decode(ids, skip_special_tokens=False)

    @cached_property
    def _config(self) -> dict:
        return json.loads((self.model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))

    def apply_chat_template(self, messages: list[dict], *, add_generation_prompt: bool) -> str:
        """Render the model's own chat template with the settings `transformers` uses:
        a sandboxed environment, trim_blocks and lstrip_blocks, and `raise_exception`."""
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message):
            raise TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        template = env.from_string(self._config["chat_template"])
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                               bos_token=self._config.get("bos_token") or "",
                               eos_token=self._config.get("eos_token") or "")
