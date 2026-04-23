"""Test tool result messages with custom_content attachments."""

import base64

from aidial_sdk.chat_completion import Attachment, CustomContent

from aidial_adapter_anthropic.dial._message import HumanToolResultMessage


async def test_tool_result_message_with_custom_content(adapter):
    """Test that tool result messages with custom_content are properly converted."""

    # Create a sample image attachment
    image_base64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("utf-8")

    # Create a tool result message with custom_content
    custom_content = CustomContent(
        attachments=[Attachment(type="image/png", data=image_base64)]
    )

    tool_result_message = HumanToolResultMessage(
        id="call_123",
        content="Tool execution result",
        custom_content=custom_content,
    )

    # Convert to DIAL message and back to verify round-trip works
    dial_message = tool_result_message.to_message()
    assert dial_message.role.value == "tool"
    assert dial_message.tool_call_id == "call_123"
    assert dial_message.content == "Tool execution result"
    assert dial_message.custom_content == custom_content

    # Verify we can reconstruct the message from DIAL format
    reconstructed = HumanToolResultMessage.from_message(dial_message)
    assert reconstructed is not None
    assert reconstructed.id == "call_123"
    assert reconstructed.content == "Tool execution result"
    assert reconstructed.custom_content == custom_content
    assert len(reconstructed.attachments) == 1
    assert reconstructed.attachments[0].type == "image/png"


async def test_tool_result_message_without_custom_content(adapter):
    """Test that tool result messages without custom_content still work."""

    # Create a tool result message without custom_content
    tool_result_message = HumanToolResultMessage(
        id="call_456",
        content="Tool execution result without attachments",
    )

    # Convert to DIAL message and back to verify it works
    dial_message = tool_result_message.to_message()
    assert dial_message.role.value == "tool"
    assert dial_message.tool_call_id == "call_456"
    assert dial_message.content == "Tool execution result without attachments"
    assert dial_message.custom_content is None

    # Verify we can reconstruct the message from DIAL format
    reconstructed = HumanToolResultMessage.from_message(dial_message)
    assert reconstructed is not None
    assert reconstructed.id == "call_456"
    assert reconstructed.content == "Tool execution result without attachments"
    assert reconstructed.custom_content is None
    assert len(reconstructed.attachments) == 0


async def test_tool_result_message_attachments_property(adapter):
    """Test the attachments property of HumanToolResultMessage."""
    # Test with custom_content
    image_base64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("utf-8")
    custom_content = CustomContent(
        attachments=[
            Attachment(type="image/png", data=image_base64),
            Attachment(type="image/jpeg", data=image_base64),
        ]
    )

    tool_result = HumanToolResultMessage(
        id="call_789",
        content="Result",
        custom_content=custom_content,
    )

    assert len(tool_result.attachments) == 2
    assert tool_result.attachments[0].type == "image/png"
    assert tool_result.attachments[1].type == "image/jpeg"

    # Test without custom_content
    tool_result_no_content = HumanToolResultMessage(
        id="call_000",
        content="Result",
    )

    assert len(tool_result_no_content.attachments) == 0
