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

_PNG_ATTACHMENT = Attachment(type="image/png", data="AA==")


@pytest.fixture
def image_handlers() -> AttachmentProcessors:
    return AttachmentProcessors(
        text_handler=create_text_block,
        attachment_processors=[IMAGE_ATTACHMENT_PROCESSOR],
        file_storage=None,
    )


async def test_tool_result_round_trip_preserves_custom_content():
    original = HumanToolResultMessage(
        id="tool-call-id1",
        content="tool result",
        custom_content=CustomContent(attachments=[_PNG_ATTACHMENT]),
    )
    dial = original.to_message()
    restored = parse_dial_message(dial)
    assert isinstance(restored, HumanToolResultMessage)
    assert restored.id == original.id
    assert restored.content == original.content
    assert restored.custom_content is not None
    assert restored.custom_content.attachments == [_PNG_ATTACHMENT]


async def test_tool_result_text_and_image(image_handlers: AttachmentProcessors):
    msg = HumanToolResultMessage(
        id="tool-call-id1",
        content="tool result",
        custom_content=CustomContent(attachments=[_PNG_ATTACHMENT]),
    )
    system, claude_msgs = await to_claude_messages(image_handlers, [msg])
    assert system == []
    assert claude_msgs.raw_list[0].payload == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tool-call-id1",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "data": _PNG_ATTACHMENT.data,
                            "media_type": _PNG_ATTACHMENT.type,
                            "type": "base64",
                        },
                    },
                    {"type": "text", "text": "tool result"},
                ],
            },
        ],
    }


async def test_tool_result_text_only(image_handlers: AttachmentProcessors):
    msg = HumanToolResultMessage(id="tool-call-id1", content="tool result")
    system, claude_msgs = await to_claude_messages(image_handlers, [msg])
    assert system == []
    assert claude_msgs.raw_list[0].payload == {
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tool-call-id1",
                "content": [
                    {"type": "text", "text": "tool result"},
                ],
            },
        ],
        "role": "user",
    }
