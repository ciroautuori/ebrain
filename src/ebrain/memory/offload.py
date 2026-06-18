"""Symbolic Offload — compress tool logs into compact symbols for context saving.

Inspired by TencentDB's offload module: converts verbose tool output (JSON blobs,
error stacks, large data tables) into compact Mermaid/ASCII representations
that preserve semantic meaning while saving tokens.

Used by: converse.py before sending context to LLM.
"""

from __future__ import annotations

from typing import Any

# Maximum chars for tool result before offloading
MAX_TOOL_RESULT_CHARS = 2000

# Symbols for common tool result patterns
_SYMBOL_TABLE: dict[str, str] = {
    "error": "[ERR]",
    "success": "[OK]",
    "not_found": "[404]",
    "timeout": "[TIMEOUT]",
    "empty": "[EMPTY]",
}


def offload_tool_result(result: Any, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Compress a tool result for context injection.

    Strategies (in order):
    1. If it's a dict with known patterns → extract key fields
    2. If it's a list → show count + first N items
    3. If it's a string → truncate + add summary
    4. Fallback: return str representation within limit
    """
    if result is None:
        return "[EMPTY]"

    if isinstance(result, dict):
        return _offload_dict(result, max_chars)

    if isinstance(result, list):
        return _offload_list(result, max_chars)

    text = str(result)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _offload_dict(d: dict, max_chars: int) -> str:
    """Compress a dict result."""
    # Check for error pattern
    if "error" in d or "Error" in d:
        err = d.get("error") or d.get("Error")
        return f"[ERR] {str(err)[:200]}"

    # Check for common API response patterns
    keys = list(d.keys())
    if "status" in d:
        status = d["status"]
        extra = []
        for k in ("id", "name", "title", "message", "count", "total"):
            if k in d and k != "status":
                extra.append(f"{k}={d[k]}")
        extra_str = ", ".join(extra[:5])
        return f"[{status}] {extra_str}" if extra_str else f"[{status}]"

    # Generic: show keys and value types
    summary = ", ".join(f"{k}={_summarize_value(d[k])}" for k in keys[:8])
    if len(keys) > 8:
        summary += f" (+{len(keys) - 8} more)"
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


def _offload_list(lst: list, max_chars: int) -> str:
    """Compress a list result."""
    count = len(lst)
    if count == 0:
        return "[EMPTY]"

    if all(isinstance(item, dict) for item in lst[:3]):
        # List of dicts — show count + first item keys
        sample_keys = list(lst[0].keys())[:5]
        return f"[{count} items] keys: {', '.join(sample_keys)}"

    if all(isinstance(item, (str, int, float)) for item in lst[:5]):
        items_str = ", ".join(str(x) for x in lst[:5])
        suffix = f" (+{count - 5} more)" if count > 5 else ""
        result = f"[{count}] {items_str}{suffix}"
        if len(result) > max_chars:
            result = result[: max_chars - 3] + "..."
        return result

    return f"[{count} items]"


def _summarize_value(v: Any) -> str:
    """Summarize a single dict value for compact display."""
    if v is None:
        return "null"
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, str):
        if len(v) > 40:
            return v[:37] + "..."
        return v
    if isinstance(v, list):
        return f"[{len(v)}]"
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"
    return str(type(v).__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


def compress_context(
    messages: list[dict[str, str]],
    max_total_tokens: int = 8000,
) -> list[dict[str, str]]:
    """Compress a conversation context for LLM injection.

    Strategy:
    - Keep system message intact
    - Keep last N messages intact (sliding window)
    - Offload tool results in middle messages
    - Truncate long content with summary
    """
    if not messages:
        return messages

    total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    if total_tokens <= max_total_tokens:
        return messages

    compressed: list[dict[str, str]] = []
    # Always keep system + last 4 messages
    keep_last = 4
    budget = max_total_tokens

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            compressed.append(msg)
            budget -= estimate_tokens(content)
            continue

        if i >= len(messages) - keep_last:
            compressed.append(msg)
            budget -= estimate_tokens(content)
            continue

        # Middle messages: offload tool results, truncate user content
        if role == "tool" or role == "assistant":
            offloaded = offload_tool_result(content)
            compressed.append({"role": role, "content": offloaded})
            budget -= estimate_tokens(offloaded)
        else:
            if estimate_tokens(content) > budget // 2:
                content = content[: budget * 2] + "...[truncated]"
            compressed.append({"role": role, "content": content})
            budget -= estimate_tokens(content)

    return compressed
