import logging

import pytest
from aidial_sdk.chat_completion import CustomContent

from aidial_adapter_anthropic.adapter._claude.state import (
    get_message_content_from_state,
)
from aidial_adapter_anthropic.dial._message import AIRegularMessage


def _message(state: dict | None) -> AIRegularMessage:
    return AIRegularMessage(
        content="hello",
        custom_content=CustomContent(state=state) if state else None,
    )


def test_no_state():
    assert get_message_content_from_state(0, _message(None)) is None


def test_valid_state():
    state = {
        "claude_message_content": [{"type": "text", "text": "hi"}],
    }
    assert get_message_content_from_state(0, _message(state)) == [
        {"type": "text", "text": "hi"}
    ]


def test_foreign_state(caplog: pytest.LogCaptureFixture):
    state = {"gemini_message_content": [{"text": "hi"}]}

    with caplog.at_level(logging.ERROR):
        assert get_message_content_from_state(0, _message(state)) is None

    assert caplog.records == []


def test_invalid_claude_state(caplog: pytest.LogCaptureFixture):
    state = {"claude_message_content": [{"type": "no-such-block"}]}

    with caplog.at_level(logging.ERROR):
        assert get_message_content_from_state(6, _message(state)) is None

    assert (
        "Invalid state at the path 'messages[6].custom_content.state'"
        in caplog.text
    )
