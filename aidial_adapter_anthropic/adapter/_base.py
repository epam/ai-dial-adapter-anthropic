from abc import ABC, abstractmethod
from typing import Any

from aidial_sdk.chat_completion import Message
from pydantic import BaseModel

from aidial_adapter_anthropic._utils.list import ListProjection
from aidial_adapter_anthropic.adapter._errors import ValidationError
from aidial_adapter_anthropic.adapter._truncate_prompt import DiscardedMessages
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import (
    ModelParameters,
    collect_text_content,
    is_system_role,
)


class ChatCompletionAdapter(ABC):
    @abstractmethod
    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: list[Message],
    ) -> None:
        pass

    async def configuration(self) -> type[BaseModel]:
        raise NotImplementedError

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: list[Message]
    ) -> int:
        raise NotImplementedError

    async def count_completion_tokens(self, string: str) -> int:
        raise NotImplementedError

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: list[Message]
    ) -> DiscardedMessages | None:
        """
        The method truncates the list of messages to fit
        into the token limit set in `params.max_prompt_tokens`.

        If the limit isn't provided, then it returns None.
        Otherwise, returns the indices of _discarded_ messages which should be
        removed from the list to make the rest fit into the token limit.
        """
        raise NotImplementedError


def default_preprocess_messages(
    messages: list[Message],
) -> ListProjection[Message]:
    def _is_empty_system_message(msg: Message) -> bool:
        return (
            is_system_role(msg.role)
            and collect_text_content(msg.content).strip() == ""
        )

    ret: list[tuple[Message, set[int]]] = []
    idx: set[int] = set()

    for i, msg in enumerate(messages):
        idx.add(i)
        if _is_empty_system_message(msg):
            continue
        ret.append((msg, idx))
        idx = set()

    if len(ret) == 0:
        raise ValidationError("List of messages must not be empty")

    return ListProjection(ret)


def keep_last(messages: list[Any], idx: int) -> bool:
    return idx == len(messages) - 1


def keep_last_and_system_messages(messages: list[Message], idx: int) -> bool:
    return is_system_role(messages[idx].role) or keep_last(messages, idx)


def trivial_partitioner(messages: list[Any]) -> list[int]:
    return [1] * len(messages)


def _raw_message(message: Any) -> dict:
    match message:
        case tuple():
            return message[0].payload
        case dict():
            return message
        case BaseModel():
            return message.model_dump()
        case _:
            return {}


def _role(message: dict) -> str | None:
    role = message.get("role")
    match role:
        case str():
            return role
        case _:
            return None


def _has_content_block(message: dict, block_type: str) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False

    return any(
        isinstance(block, dict) and block.get("type") == block_type
        for block in content
    )


def _is_assistant_tool_call(message: dict) -> bool:
    if _role(message) != "assistant":
        return False

    return bool(message.get("tool_calls")) or _has_content_block(
        message, "tool_use"
    )


def _is_tool_result(message: dict) -> bool:
    role = _role(message)
    return (
        role == "tool"
        or bool(message.get("tool_result"))
        or _has_content_block(message, "tool_result")
    )


def claude_partitioner(messages: list[Any]) -> list[int]:
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
    if n == 0:
        return []

    raw_messages = [_raw_message(message) for message in messages]
    chunks: list[int] = []
    idx = 0

    while idx < n:
        current = raw_messages[idx]
        current_role = _role(current)

        if current_role == "system":
            chunks.append(1)
            idx += 1
            continue

        if (
            idx + 1 < n
            and current_role == "user"
            and _is_assistant_tool_call(raw_messages[idx + 1])
        ):
            end = idx + 1
            while end < n and _is_assistant_tool_call(raw_messages[end]):
                end += 1
                while end < n and _is_tool_result(raw_messages[end]):
                    end += 1

            if end < n and _role(raw_messages[end]) == "assistant":
                end += 1

            chunks.append(end - idx)
            idx = end
            continue

        if _is_assistant_tool_call(current):
            end = idx + 1
            while end < n and _is_tool_result(raw_messages[end]):
                end += 1

            if end < n and _role(raw_messages[end]) == "assistant":
                end += 1

            chunks.append(end - idx)
            idx = end
            continue

        size = 2 if idx + 1 < n else 1
        chunks.append(size)
        idx += size
    return chunks
