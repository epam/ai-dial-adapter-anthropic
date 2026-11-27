import pytest
from aidial_sdk.chat_completion.request import ChatCompletionRequest

from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.dial.request import AdapterRequest

DEFAULT_MAX_TOKENS = 1024  # see the `adapter` fixture


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"max_completion_tokens": 2048}, 2048),
        ({"max_tokens": 300}, 300),
        ({"max_tokens": 300, "max_completion_tokens": 2048}, 2048),
        ({}, DEFAULT_MAX_TOKENS),
    ],
    ids=["max_completion_tokens", "max_tokens", "both", "neither"],
)
async def test_max_tokens(adapter: Adapter, params: dict, expected: int):
    request = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hello"}], **params}
    )

    claude_request = await adapter._prepare_claude_request(
        AdapterRequest.create(request)
    )

    assert claude_request.params["max_tokens"] == expected
