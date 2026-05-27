import math
from typing import Literal

import anthropic
import pytest
from aidial_sdk.chat_completion import Function, Message, Tool
from aidial_sdk.exceptions import HTTPException as DialException
from typing_extensions import override

from aidial_adapter_anthropic.adapter import ChatCompletionAdapter
from aidial_adapter_anthropic.adapter._claude.tokenizer.approximate import (
    ApproximateTokenizer,
)
from aidial_adapter_anthropic.adapter._truncate_prompt import DiscardedMessages
from aidial_adapter_anthropic.adapter.claude import create_adapter
from aidial_adapter_anthropic.dial.request import ModelParameters
from aidial_adapter_anthropic.dial.tools import ToolsConfig, ToolsMode
from tests.utils.openai import (
    ai,
    ai_tool_call,
    sys,
    tool_result,
    user,
    user_with_image,
)

_TOOL_SYSTEM_MESSAGE = 55


class _MockTokenizer(ApproximateTokenizer):
    @override
    def tokenize_text(self, text: str) -> int:
        try:
            return int(text)
        except Exception:
            return 1

    @override
    def tokenize_tool_system_message(
        self, tool_choice: Literal["none", "auto", "any", "tool"]
    ) -> int:
        return _TOOL_SYSTEM_MESSAGE


@pytest.fixture
async def model():
    return await create_adapter(
        deployment="test-anthropic-deployment",
        storage=None,
        client=anthropic.AsyncAnthropic(),
        custom_tokenizer=_MockTokenizer(),
        default_max_tokens=1024,
        supports_thinking=True,
        supports_documents=True,
    )


async def tokenize(
    model: ChatCompletionAdapter,
    messages: list[Message],
    tool_config: ToolsConfig | None = None,
) -> int:
    params = ModelParameters(tool_config=tool_config)
    return await model.count_prompt_tokens(params, messages)


async def compute_discarded_messages(
    model: ChatCompletionAdapter,
    messages: list[Message],
    max_prompt_tokens: int | None,
    tool_config: ToolsConfig | None = None,
) -> DiscardedMessages | str:
    params = ModelParameters(
        max_prompt_tokens=max_prompt_tokens, tool_config=tool_config
    )

    try:
        return await model.compute_discarded_messages(params, messages) or []
    except DialException as e:
        return e.message


def _index_range(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


_TOOL_CONFIG = ToolsConfig(
    tools=[Tool(type="function", function=Function(name="function"))],
    tool_choice="auto",
    tool_ids={},
    tools_mode=ToolsMode.TOOLS,
)

_PER_MESSAGE_TOKENS = 5

_PNG_IMAGE_50_50 = "iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAIAAACRXR/mAAAAS0lEQVR4nO3OsQEAEADAMPz/Mw9YMjE0F2Tu8aP1OnBXS9QStUQtUUvUErVELVFL1BK1RC1RS9QStUQtUUvUErVELVFL1BK1RC1xAEGqAWOFuDKrAAAAAElFTkSuQmCC"

_PNG_IMAGE_50_50_TOKENS = math.ceil((50 * 50) / 750.0)


async def test_one_turn_no_truncation(model):
    messages = [
        sys("11"),
        user("22"),
        ai("33"),
    ]

    expected_tokens = (
        11 + (_PER_MESSAGE_TOKENS + 22) + (_PER_MESSAGE_TOKENS + 33)
    )

    assert await tokenize(model, messages) == expected_tokens

    discarded_messages = await compute_discarded_messages(
        model, messages, expected_tokens
    )

    assert discarded_messages == []


async def test_one_turn_with_image(model):
    messages = [
        sys("11"),
        user_with_image("22", _PNG_IMAGE_50_50),
    ]

    expected_tokens = 11 + (_PER_MESSAGE_TOKENS + _PNG_IMAGE_50_50_TOKENS + 22)

    assert await tokenize(model, messages) == expected_tokens

    truncation = await compute_discarded_messages(
        model, messages, expected_tokens
    )

    assert truncation == []

    truncation = await compute_discarded_messages(
        model, messages, expected_tokens - 1
    )

    assert (
        truncation
        == f"The requested maximum prompt tokens is {expected_tokens - 1}. However, the system messages and the last user message resulted in {expected_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


async def test_one_turn_with_tools(model):
    messages = [
        sys("11"),
        user("22"),
        ai("33"),
    ]

    expected_tokens = (
        _TOOL_SYSTEM_MESSAGE
        + 11
        + 1
        + (_PER_MESSAGE_TOKENS + 22)
        + (_PER_MESSAGE_TOKENS + 33)
    )

    assert await tokenize(model, messages, _TOOL_CONFIG) == expected_tokens

    discarded_messages = await compute_discarded_messages(
        model, messages, expected_tokens - 1, _TOOL_CONFIG
    )

    assert (
        discarded_messages
        == f"The requested maximum prompt tokens is {expected_tokens - 1}. However, the system messages and the last user message resulted in {expected_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


async def test_one_turn_overflow(model):
    messages = [
        sys("11"),
        user("22"),
        ai("33"),
    ]

    expected_tokens = (
        11 + (22 + _PER_MESSAGE_TOKENS) + (33 + _PER_MESSAGE_TOKENS)
    )

    truncation_error = await compute_discarded_messages(model, messages, 1)

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in {expected_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


async def test_multiple_system_messages(model):
    messages = [
        sys("11"),
        sys("22"),
        user("33"),
    ]

    expected_tokens = (11 + 22) + (_PER_MESSAGE_TOKENS + 33)

    assert await tokenize(model, messages) == expected_tokens


async def test_truncate_first_turn(model):
    messages = [
        user("11"),
        ai("22"),
        user("33"),
        ai("44"),
    ]

    expected_tokens = (
        (_PER_MESSAGE_TOKENS + 11)
        + (_PER_MESSAGE_TOKENS + 22)
        + (_PER_MESSAGE_TOKENS + 33)
        + (_PER_MESSAGE_TOKENS + 44)
    )

    assert await tokenize(model, messages) == expected_tokens

    discarded_messages = await compute_discarded_messages(
        model, messages, (_PER_MESSAGE_TOKENS + 33) + (_PER_MESSAGE_TOKENS + 44)
    )

    assert discarded_messages == [0, 1]


async def test_truncate_first_turn_with_system(model):
    messages = [
        sys("11"),
        user("22"),
        ai("33"),
        user("44"),
        ai("55"),
    ]

    discarded_messages = await compute_discarded_messages(
        model,
        messages,
        11 + (_PER_MESSAGE_TOKENS + 44) + (_PER_MESSAGE_TOKENS + 55),
    )

    assert discarded_messages == [1, 2]


async def test_truncate_first_turn_with_system_2(model):
    # Equivalent of test_truncate_first_turn_with_system with adjacent messages with the same role
    messages = [
        sys("11"),
        # 22 block (discarded)
        user("10"),
        user("12"),
        # 33 block (discarded)
        ai("10"),
        ai("11"),
        ai("12"),
        # 44 block
        user("44"),
        # 55 block
        ai("12"),
        ai("22"),
        ai("21"),
    ]

    discarded_messages = await compute_discarded_messages(
        model,
        messages,
        11 + (_PER_MESSAGE_TOKENS + 44) + (_PER_MESSAGE_TOKENS + 55),
    )

    assert discarded_messages == [1, 2, 3, 4, 5]


async def test_truncate_first_turn_with_system_3(model):
    # Equivalent of test_truncate_first_turn_with_system_2 with one less tokens requests than the critical amount
    messages = [
        sys("11"),
        # 22 block (discarded)
        user("10"),
        user("12"),
        # 33 block (discarded)
        ai("10"),
        ai("11"),
        ai("12"),
        # 44 block
        user("44"),
        # 55 block
        ai("12"),
        ai("22"),
        ai("21"),
    ]

    min_possible_tokens = (
        11 + (_PER_MESSAGE_TOKENS + 44) + (_PER_MESSAGE_TOKENS + 55)
    )

    truncation_error = await compute_discarded_messages(
        model,
        messages,
        min_possible_tokens - 1,
    )

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is {min_possible_tokens - 1}. However, the system messages and the last user message resulted in {min_possible_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


async def test_zero_turn_overflow(model):
    messages = [
        sys("11"),
        user("22"),
    ]

    expected_tokens = 11 + (22 + _PER_MESSAGE_TOKENS)

    truncation_error = await compute_discarded_messages(model, messages, 3)

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is 3. However, the system messages and the last user message resulted in {expected_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


async def test_chat_history_overflow(model):
    messages = [
        sys("11"),
        user("22"),
        ai("33"),
        user("44"),
    ]

    min_possible_tokens = 11 + (44 + _PER_MESSAGE_TOKENS)

    truncation_error = await compute_discarded_messages(model, messages, 1)

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in {min_possible_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


async def test_chat_history_overflow_2(model):
    # Equivalent of test_chat_history_overflow with adjacent messages with the same role
    messages = [
        sys("11"),
        user("11"),
        user("11"),
        ai("11"),
        ai("22"),
        user("44"),
    ]

    min_possible_tokens = 11 + (44 + _PER_MESSAGE_TOKENS)

    truncation_error = await compute_discarded_messages(model, messages, 1)

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in {min_possible_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.parametrize(
    ("max_prompt_tokens", "expected_discarded"),
    [
        # minimal feasible prompt:
        # system(1) + last_user(5 + 1) = 7
        # => trunc block_1, intermediate messages and block_2
        (7, _index_range(1, 12)),
        (32, _index_range(1, 12)),
        # trunc block_1 + intermediate msgs:
        # system(1) + block_2(user=6 + tool_call=7 + tool_result=7 + ai=6)
        # + last_user(6) = 33.
        (33, _index_range(1, 8)),
        (44, _index_range(1, 8)),
        # trunc only block_1:
        # full_prompt(85) - block_1(40) = 45
        (45, _index_range(1, 6)),
        (46, _index_range(1, 6)),
        # full prompt / no trunc
        # system(1) + 13 messages * 5 + content(19) = 85
        (85, []),
    ],
)
async def test_truncate_tool_call_cascade(
    model, max_prompt_tokens: int, expected_discarded: list[int]
):
    messages = [
        sys("system"),
        # <block_1>
        user("user"),
        ai_tool_call("tool_1"),
        tool_result("tool_1"),
        ai_tool_call("tool_2"),
        tool_result("tool_2"),
        ai("ai"),
        # </block_1>
        user("user"),
        ai("ai"),
        # <block_2>
        user("user"),
        ai_tool_call("tool_3"),
        tool_result("tool_3"),
        ai("ai"),
        # </block_2>
        user("user"),
    ]
    discarded_messages = await compute_discarded_messages(
        model=model, messages=messages, max_prompt_tokens=max_prompt_tokens
    )

    assert discarded_messages == expected_discarded
