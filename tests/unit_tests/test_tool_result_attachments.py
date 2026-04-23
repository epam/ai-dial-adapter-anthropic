from typing import Any, cast

import pytest
from aidial_sdk.chat_completion import Attachment, CustomContent

from aidial_adapter_anthropic.adapter._claude.blocks import (
    IMAGE_ATTACHMENT_PROCESSOR,
    create_text_block,
)
from aidial_adapter_anthropic.adapter._claude.converters import (
    to_claude_messages,
)
from aidial_adapter_anthropic.dial._attachments import AttachmentProcessors
from aidial_adapter_anthropic.dial._message import (
    HumanToolResultMessage,
    parse_dial_message,
)

_PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def image_handlers() -> AttachmentProcessors:
    return AttachmentProcessors(
        text_handler=create_text_block,
        attachment_processors=[IMAGE_ATTACHMENT_PROCESSOR],
        file_storage=None,
    )


async def test_tool_result_round_trip_preserves_custom_content():
    original = HumanToolResultMessage(
        id="toolu_01",
        content="Result text",
        custom_content=CustomContent(
            attachments=[Attachment(type="image/png", data=_PNG_1X1)]
        ),
    )
    dial = original.to_message()
    restored = parse_dial_message(dial)
    assert isinstance(restored, HumanToolResultMessage)
    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.custom_content is not None
    assert restored.custom_content.attachments is not None
    assert len(restored.custom_content.attachments) == 1
    att = restored.custom_content.attachments[0]
    assert att.type == "image/png"
    assert att.data == _PNG_1X1


async def test_tool_result_to_claude_includes_image_and_text(
    image_handlers: AttachmentProcessors,
):
    msg = HumanToolResultMessage(
        id="call_1",
        content="see screenshot",
        custom_content=CustomContent(
            attachments=[Attachment(type="image/png", data=_PNG_1X1)]
        ),
    )
    _system, claude_msgs = await to_claude_messages(image_handlers, [msg])
    projected = claude_msgs.raw_list
    assert len(projected) == 1
    blocks_raw = projected[0].payload["content"]
    assert isinstance(blocks_raw, list)
    assert len(blocks_raw) == 1
    tool_result = cast(Any, blocks_raw[0])
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"
    inner_raw = tool_result["content"]
    assert isinstance(inner_raw, list)
    assert len(inner_raw) == 2
    assert inner_raw[0]["type"] == "image"
    assert inner_raw[1]["type"] == "text"
    assert inner_raw[1]["text"] == "see screenshot"


async def test_tool_result_text_only_backward_compatible(
    image_handlers: AttachmentProcessors,
):
    msg = HumanToolResultMessage(id="call_2", content="plain")
    _system, claude_msgs = await to_claude_messages(image_handlers, [msg])
    blocks_raw = claude_msgs.raw_list[0].payload["content"]
    assert isinstance(blocks_raw, list)
    tool_result = cast(Any, blocks_raw[0])
    tr_content = tool_result["content"]
    assert isinstance(tr_content, list)
    assert tr_content == [create_text_block("plain")]
