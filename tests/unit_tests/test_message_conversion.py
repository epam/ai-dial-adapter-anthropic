from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.dial.request import ModelParameters
from tests.utils.openai import ai, sys, user


async def test_empty_message_is_replaced_with_space(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(), [user("")]
    )
    assert request.claude_messages == [
        {"role": "user", "content": [{"text": " ", "type": "text"}]}
    ]


async def test_assistant_message_is_replaced_with_space(adapter: Adapter):
    request = await adapter._prepare_claude_request(ModelParameters(), [ai("")])
    assert request.claude_messages == [
        {"role": "assistant", "content": [{"text": " ", "type": "text"}]}
    ]


async def test_empty_system_message_is_removed(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(), [sys(""), user("hello")]
    )
    assert request.claude_messages == [
        {"role": "user", "content": [{"text": "hello", "type": "text"}]}
    ]


async def test_mid_conversation_system_message(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        ModelParameters(),
        [sys("top"), user("hi"), sys("mid"), ai("last")],
    )

    assert request.params["system"] == [{"text": "top", "type": "text"}]
    assert request.claude_messages == [
        {"role": "user", "content": [{"text": "hi", "type": "text"}]},
        {"role": "system", "content": [{"text": "mid", "type": "text"}]},
        {"role": "assistant", "content": [{"text": "last", "type": "text"}]},
    ]


async def test_consecutive_mid_conversation_system_messages_are_merged(
    adapter: Adapter,
):
    request = await adapter._prepare_claude_request(
        ModelParameters(),
        [user("hi"), sys("mid1"), sys("mid2"), ai("last")],
    )

    assert request.claude_messages == [
        {"role": "user", "content": [{"text": "hi", "type": "text"}]},
        {
            "role": "system",
            "content": [
                {"text": "mid1", "type": "text"},
                {"text": "mid2", "type": "text"},
            ],
        },
        {"role": "assistant", "content": [{"text": "last", "type": "text"}]},
    ]
