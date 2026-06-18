"""EBrain LLM — injectable LLM interface for memory extraction and knowledge synthesis.

Usage:
    from ebrain.llm import set_ask_json, ask_json

    # Set your LLM function
    async def my_ask_json(prompt: str, **kwargs) -> dict:
        ...

    set_ask_json(my_ask_json)

This module is a SEAM: inject any LLM backend (Claude, OpenAI, local) without
ebrain depending on a specific provider. Default raises NotImplementedError.
"""

from __future__ import annotations

from typing import Any
from typing import Awaitable
from typing import Callable

# Type for an LLM function that takes a prompt and returns JSON
AskJSON = Callable[[str], Awaitable[dict[str, Any]]]

_ask_json: AskJSON | None = None
_default_model: str | None = None


def set_ask_json(fn: AskJSON) -> None:
    """Inject an LLM function. Must be async and return a dict."""
    global _ask_json
    _ask_json = fn


def set_default_model(model: str) -> None:
    """Set the default model identifier."""
    global _default_model
    _default_model = model


async def ask_json(prompt: str, *, model: str | None = None) -> dict[str, Any]:
    """Call the injected LLM. Raises RuntimeError if not configured."""
    if _ask_json is None:
        raise RuntimeError(
            "ebrain LLM not configured. Call ebrain.llm.set_ask_json(your_llm_function) first."
        )
    return await _ask_json(prompt)


def get_default_model() -> str:
    return _default_model or "unknown"


def is_configured() -> bool:
    return _ask_json is not None
