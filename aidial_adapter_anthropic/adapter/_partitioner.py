from typing import Any, Literal

from anthropic.types.beta import BetaMessageParam

from aidial_adapter_anthropic.adapter._base import keep_last
from aidial_adapter_anthropic.dial._attachments import WithResources

ClaudeMessages = list[tuple[WithResources[BetaMessageParam], set[int]]]


def _has_content_block(message: BetaMessageParam, block_type: str) -> bool:
    content = message["content"]
    if isinstance(content, str):
        return False

    return any(
        isinstance(block, dict) and block.get("type") == block_type
        for block in content
    )


def _is_assistant_tool_call(message: BetaMessageParam) -> bool:
    return _role(message) == "assistant" and _has_content_block(
        message, "tool_use"
    )


def _is_tool_result(message: BetaMessageParam) -> bool:
    return _role(message) == "user" and _has_content_block(
        message, "tool_result"
    )


def _role(message: BetaMessageParam) -> Literal["user", "assistant", "system"]:
    return message["role"]


def claude_partitioner(messages: ClaudeMessages) -> list[int]:
    """
    Build truncation partitions for Claude history.

    Messages in the same partition are removed/kept atomically by the
    truncation algorithm.

    Partitioning rules:
    - Default behavior follows turn-based truncation (pairs of two).
    - Tool-call flows are grouped as transactions:
      `user* -> (assistant(tool_call) | tool_result)* -> assistant*`.
      This prevents orphan tool-result blocks when earlier history is dropped.
    - A mid-conversation system message sits between the tool/user part and
      the assistant reply, so it is absorbed into its turn and kept or dropped
      atomically with it. This keeps system placement valid after truncation.
    """
    n = len(messages)
    unwrapped = [m[0].payload for m in messages]
    ret: list[int] = []
    idx = 0

    while idx < n:
        end = idx
        while end < n and _role(unwrapped[end]) == "user":
            end += 1
        while end < n and (
            _is_assistant_tool_call(unwrapped[end])
            or _is_tool_result(unwrapped[end])
        ):
            end += 1
        while end < n and _role(unwrapped[end]) == "system":
            end += 1
        while end < n and _role(unwrapped[end]) == "assistant":
            end += 1

        ret.append(end - idx)
        idx = end

    return ret


def keep_last_or_system(messages: ClaudeMessages, idx: int) -> bool:
    """
    Keep the last message and every mid-conversation system message.

    System messages carry operator-level instructions that must survive
    truncation; their turn is kept atomically (see `claude_partitioner`).
    """
    return _role(messages[idx][0].payload) == "system" or keep_last(
        messages, idx
    )


def trivial_partitioner(messages: list[Any]) -> list[int]:
    return [1] * len(messages)
