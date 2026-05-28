from typing import Any, Literal

from anthropic.types.beta import BetaMessageParam

from aidial_adapter_anthropic.dial._attachments import WithResources


def _has_content_block(message: BetaMessageParam, block_type: str) -> bool:
    content = message["content"]
    if isinstance(content, str):
        return False

    return any(
        isinstance(block, dict) and block.get("type") == block_type
        for block in content
    )


def _is_assistant_tool_call(payload: BetaMessageParam) -> bool:
    if payload["role"] != "assistant":
        return False

    return _has_content_block(payload, "tool_use")


def _is_tool_result(payload: BetaMessageParam) -> bool:
    return payload["role"] == "tool" or _has_content_block(
        payload, "tool_result"
    )


def _role(payload: BetaMessageParam) -> Literal["user", "assistant"]:
    return payload["role"]


def _payload(
    i: tuple[WithResources[BetaMessageParam], set[int]],
) -> BetaMessageParam:
    return i[0].payload


def claude_partitioner(
    messages: list[tuple[WithResources[BetaMessageParam], set[int]]],
) -> list[int]:
    """
    Build truncation partitions for Claude history.

    Messages in the same partition are removed/kept atomically by the
    truncation algorithm.

    Partitioning rules:
    - Default behavior follows turn-based truncation (pairs of two).
    - Tool-call flows are grouped as transactions:
      `user -> assistant(tool_call)+ -> tool_result+ -> assistant?`.
      This prevents orphan tool-result blocks when earlier history is dropped.
    """
    n = len(messages)
    payloads = [_payload(msg) for msg in messages]
    ret: list[int] = []
    idx = 0

    while idx < n:
        end = idx
        while end < n and _role(payloads[end]) == "user":
            end += 1
        while end < n and (
            _is_assistant_tool_call(payloads[end])
            or _is_tool_result(payloads[end])
        ):
            end += 1
        while end < n and _role(payloads[end]) == "assistant":
            end += 1

        ret.append(end - idx)
        idx = end

    return ret


def trivial_partitioner(messages: list[Any]) -> list[int]:
    return [1] * len(messages)
