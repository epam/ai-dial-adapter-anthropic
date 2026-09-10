from aidial_sdk.chat_completion import Attachment
from anthropic.types.beta import BetaServerToolUseBlock as ServerToolUseBlock
from anthropic.types.beta import (
    BetaWebSearchResultBlock as WebSearchResultBlock,
)
from anthropic.types.beta import (
    BetaWebSearchToolResultBlock as WebSearchToolResultBlock,
)
from anthropic.types.beta import (
    BetaWebSearchToolResultError as WebSearchToolResultError,
)

from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from tests.utils.consumer import invoke_non_streaming


async def test_web_search_server_tool_creates_web_search_stage(
    adapter: Adapter,
):
    consumer = await invoke_non_streaming(
        adapter,
        [
            ServerToolUseBlock(
                type="server_tool_use",
                id="srv_1",
                name="web_search",
                input={"query": "weather in NYC"},
            )
        ],
    )

    assert consumer.stages["Web Search"] == ["weather in NYC"]


async def test_web_search_result_creates_attachments(adapter: Adapter):
    consumer = await invoke_non_streaming(
        adapter,
        [
            WebSearchToolResultBlock(
                type="web_search_tool_result",
                tool_use_id="srv_1",
                content=[
                    WebSearchResultBlock(
                        type="web_search_result",
                        title="Example",
                        url="https://example.com",
                        encrypted_content="ZW5jcnlwdGVk",
                        page_age="1 day ago",
                    )
                ],
            )
        ],
    )

    assert consumer.attachments == [
        Attachment(title="Example", url="https://example.com")
    ]


async def test_web_search_result_error_reported_in_stage(
    adapter: Adapter,
):
    consumer = await invoke_non_streaming(
        adapter,
        [
            WebSearchToolResultBlock(
                type="web_search_tool_result",
                tool_use_id="srv_1",
                content=WebSearchToolResultError(
                    type="web_search_tool_result_error",
                    error_code="max_uses_exceeded",
                ),
            )
        ],
    )

    assert consumer.stages["Web Search"] == [
        "Web search failed: max_uses_exceeded"
    ]


async def test_server_tool_use_persists_state_without_thinking(
    adapter: Adapter,
):
    adapter.supports_thinking = False
    consumer = await invoke_non_streaming(
        adapter,
        [
            ServerToolUseBlock(
                type="server_tool_use",
                id="srv_1",
                name="web_search",
                input={"query": "latest AI news"},
            )
        ],
    )

    assert consumer._choice.state is not None


async def test_plain_text_does_not_persist_state_without_thinking(
    adapter: Adapter,
):
    adapter.supports_thinking = False
    consumer = await invoke_non_streaming(adapter, [])

    assert consumer._choice.state is None
