import base64

import pytest
from aidial_sdk.chat_completion import (
    Attachment,
    CacheBreakpoint,
    CustomContent,
    MessageContentTextPart,
)

from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.adapter._errors import ValidationError
from aidial_adapter_anthropic.dial._message import AIRegularMessage
from aidial_adapter_anthropic.dial.request import AdapterRequest
from tests.utils.openai import ai, prompt, sys, user

_THINKING_BLOCK = {"type": "thinking", "thinking": "hmm", "signature": "sig"}


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def _ai_with_custom_content(**kwargs) -> AIRegularMessage:
    return AIRegularMessage(content="", custom_content=CustomContent(**kwargs))


async def test_empty_message_is_replaced_with_space(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        AdapterRequest(messages=prompt(user("")))
    )
    assert request.claude_messages == [
        {"role": "user", "content": [{"text": " ", "type": "text"}]}
    ]


async def test_only_empty_assistant_message_is_rejected(adapter: Adapter):
    with pytest.raises(
        ValidationError, match="Messages must contain non-whitespace text"
    ):
        await adapter._prepare_claude_request(
            AdapterRequest(messages=prompt(ai("")))
        )


async def test_empty_system_message_is_removed(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        AdapterRequest(messages=prompt(sys(""), user("hello")))
    )
    assert request.claude_messages == [
        {"role": "user", "content": [{"text": "hello", "type": "text"}]}
    ]


async def test_mid_conversation_system_message(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        AdapterRequest(
            messages=prompt(sys("top"), user("hi"), sys("mid"), ai("last"))
        )
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
        AdapterRequest(
            messages=prompt(user("hi"), sys("mid1"), sys("mid2"), ai("last"))
        )
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


async def test_empty_assistant_message_between_user_messages_is_dropped(
    adapter: Adapter,
):
    request = await adapter._prepare_claude_request(
        AdapterRequest(messages=prompt(user("Hi"), ai(""), user("Hi")))
    )
    assert request.claude_messages == [
        {"role": "user", "content": [_text("Hi"), _text("Hi")]}
    ]
    assert request.messages.lst[0][1] == {0, 1, 2}


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(" ", id="space"),
        pytest.param("\n\t", id="whitespace"),
        pytest.param([], id="no-parts"),
        pytest.param(
            [MessageContentTextPart(type="text", text="")], id="empty-text-part"
        ),
        pytest.param(
            [MessageContentTextPart(type="text", text=" ")],
            id="blank-text-part",
        ),
    ],
)
async def test_blank_assistant_message_is_dropped(adapter: Adapter, content):
    request = await adapter._prepare_claude_request(
        AdapterRequest(
            messages=prompt(
                user("Hi"), AIRegularMessage(content=content), user("Hi")
            )
        )
    )
    assert request.claude_messages == [
        {"role": "user", "content": [_text("Hi"), _text("Hi")]}
    ]
    assert request.messages.lst[0][1] == {0, 1, 2}


async def test_leading_empty_assistant_message_is_dropped(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        AdapterRequest(messages=prompt(ai(""), user("Hi")))
    )
    assert request.claude_messages == [
        {"role": "user", "content": [_text("Hi")]}
    ]
    assert request.messages.lst[0][1] == {0, 1}


async def test_trailing_empty_assistant_message_is_dropped(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        AdapterRequest(messages=prompt(user("Hi"), ai("")))
    )
    assert request.claude_messages == [
        {"role": "user", "content": [_text("Hi")]}
    ]
    assert request.messages.lst[0][1] == {0, 1}


async def test_multiple_empty_assistant_messages_are_dropped(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        AdapterRequest(
            messages=prompt(user("1"), ai(""), user("2"), ai(""), user("3"))
        )
    )
    assert request.claude_messages == [
        {"role": "user", "content": [_text("1"), _text("2"), _text("3")]}
    ]
    assert request.messages.lst[0][1] == {0, 1, 2, 3, 4}


async def test_empty_assistant_message_next_to_non_empty_one(adapter: Adapter):
    request = await adapter._prepare_claude_request(
        AdapterRequest(
            messages=prompt(user("Hi"), ai(""), ai("Hello"), user("Bye"))
        )
    )
    assert request.claude_messages == [
        {"role": "user", "content": [_text("Hi")]},
        {"role": "assistant", "content": [_text("Hello")]},
        {"role": "user", "content": [_text("Bye")]},
    ]
    assert [indices for _, indices in request.messages.lst] == [
        {0},
        {1, 2},
        {3},
    ]


async def test_dropped_assistant_message_with_cache_breakpoint(
    adapter: Adapter,
):
    message = AIRegularMessage(content="", cache_breakpoint=CacheBreakpoint())
    request = await adapter._prepare_claude_request(
        AdapterRequest(messages=prompt(user("Hi"), message, user("Hi")))
    )
    assert request.claude_messages == [
        {"role": "user", "content": [_text("Hi"), _text("Hi")]}
    ]


@pytest.mark.parametrize(
    "state_content",
    [
        pytest.param([_THINKING_BLOCK, _text("Hello")], id="thinking-and-text"),
        pytest.param([_THINKING_BLOCK], id="thinking-only"),
    ],
)
async def test_empty_assistant_message_with_claude_state_is_kept(
    adapter: Adapter, state_content: list[dict]
):
    message = _ai_with_custom_content(
        state={"claude_message_content": state_content}
    )
    request = await adapter._prepare_claude_request(
        AdapterRequest(messages=prompt(user("Hi"), message, user("Hi")))
    )
    assert request.claude_messages == [
        {"role": "user", "content": [_text("Hi")]},
        {"role": "assistant", "content": state_content},
        {"role": "user", "content": [_text("Hi")]},
    ]


@pytest.mark.parametrize(
    "state",
    [
        pytest.param({"foo": 1}, id="foreign-state"),
        pytest.param(
            {"claude_message_content": [_text("")]}, id="empty-text-state"
        ),
    ],
)
async def test_empty_assistant_message_with_blank_state_is_dropped(
    adapter: Adapter, state: dict
):
    message = _ai_with_custom_content(state=state)
    request = await adapter._prepare_claude_request(
        AdapterRequest(messages=prompt(user("Hi"), message, user("Hi")))
    )
    assert request.claude_messages == [
        {"role": "user", "content": [_text("Hi"), _text("Hi")]}
    ]
    assert request.messages.lst[0][1] == {0, 1, 2}


async def test_empty_assistant_message_with_text_attachment_is_kept(
    adapter: Adapter,
):
    data = base64.b64encode(b"document").decode()
    message = _ai_with_custom_content(
        attachments=[Attachment(type="text/plain", data=data)]
    )
    request = await adapter._prepare_claude_request(
        AdapterRequest(messages=prompt(user("Hi"), message, user("Hi")))
    )
    assert request.claude_messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "document",
                },
                "citations": {"enabled": False},
            }
        ],
    }


async def test_empty_assistant_message_with_image_attachment_is_kept(
    adapter: Adapter,
):
    # Claude rejects images in assistant turns with a clear error,
    # so the message must not be hidden by dropping it.
    message = _ai_with_custom_content(
        attachments=[Attachment(type="image/png", data="AA==")]
    )
    request = await adapter._prepare_claude_request(
        AdapterRequest(messages=prompt(user("Hi"), message, user("Hi")))
    )
    assert request.claude_messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "AA==",
                },
            }
        ],
    }
