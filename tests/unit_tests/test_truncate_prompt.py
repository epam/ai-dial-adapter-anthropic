from typing import TypedDict

from aidial_adapter_anthropic.adapter._base import (
    trivial_partitioner,
)
from aidial_adapter_anthropic.adapter._truncate_prompt import (
    DiscardedMessages,
    TruncatePromptError,
    _partition_indexer,
    compute_discarded_messages,
)


class _Message(TypedDict):
    is_system: bool
    content: str


def _msg(content: str) -> _Message:
    return {"content": content, "is_system": False}


def _sys(content: str) -> _Message:
    return {"content": content, "is_system": True}


def keep_last_and_system_messages(messages: list[_Message], idx: int) -> bool:
    return messages[idx]["is_system"] or idx == len(messages) - 1


async def truncate_prompt_by_words(
    messages: list[_Message],
    user_limit: int,
    model_limit: int | None = None,
) -> DiscardedMessages | TruncatePromptError:
    async def _tokenize_by_words(messages: list[_Message]) -> int:
        return sum(len(msg["content"].split()) for msg in messages)

    return await compute_discarded_messages(
        messages=messages,
        tokenizer=_tokenize_by_words,
        keep_message=keep_last_and_system_messages,
        partitioner=trivial_partitioner,
        model_limit=model_limit,
        user_limit=user_limit,
    )


def test_partition_indexer():
    assert [_partition_indexer([2, 3])(i) for i in range(5)] == [
        [0, 1],
        [0, 1],
        [2, 3, 4],
        [2, 3, 4],
        [2, 3, 4],
    ]


async def test_no_truncation():
    messages = [
        _sys("text1"),
        _msg("text2"),
        _msg("text3"),
    ]

    discarded_messages = await truncate_prompt_by_words(
        messages=messages, user_limit=3
    )

    assert discarded_messages == []


async def test_truncation():
    messages = [
        _sys("system1"),
        _msg("remove1"),
        _sys("system2"),
        _msg("remove2"),
        _msg("query"),
    ]
    discarded_messages = await truncate_prompt_by_words(
        messages=messages, user_limit=3
    )

    assert discarded_messages == [1, 3]


async def test_truncation_with_one_message_left():
    messages = [
        _msg("reply"),
        _msg("query"),
    ]

    discarded_messages = await truncate_prompt_by_words(
        messages=messages, user_limit=1
    )

    assert discarded_messages == [0]


async def test_truncation_with_one_message_accepted_after_second_check():
    messages = [
        _msg("hello world"),
        _msg("query"),
    ]

    discarded_messages = await truncate_prompt_by_words(
        messages=messages, user_limit=1
    )

    assert discarded_messages == [0]


async def test_prompt_is_too_big():
    messages = [
        _sys("text1"),
        _sys("text2"),
        _msg("text3"),
    ]

    truncation_error = await truncate_prompt_by_words(
        messages=messages, user_limit=2
    )

    assert (
        isinstance(truncation_error, TruncatePromptError)
        and truncation_error.print()
        == "The requested maximum prompt tokens is 2. However, the system messages and the last user message resulted in 3 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


async def test_prompt_with_history_is_too_big():
    messages = [
        _sys("text1"),
        _msg("text2"),
        _msg("text3"),
    ]

    truncation_error = await truncate_prompt_by_words(
        messages=messages, user_limit=1
    )

    assert (
        isinstance(truncation_error, TruncatePromptError)
        and truncation_error.print()
        == "The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in 2 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


async def test_inconsistent_limits():
    messages = [_msg("text2")]

    truncation_error = await truncate_prompt_by_words(
        messages=messages, user_limit=10, model_limit=5
    )

    assert (
        isinstance(truncation_error, TruncatePromptError)
        and truncation_error.print()
        == "The request maximum prompt tokens is 10. However, the model's maximum context length is 5 tokens."
    )
