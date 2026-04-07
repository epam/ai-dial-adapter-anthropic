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
