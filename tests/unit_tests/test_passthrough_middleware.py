"""Per-cloud request adaptation, parametrized over every supported backend.

Payloads are built from the Anthropic SDK's own types and sent through the SDK
client, so each test exercises the same bytes a real caller would send. What is
asserted is what reached the upstream — the header and body the backend sees
after the package's middlewares had their say.
"""

import json

import anthropic
import pytest
from anthropic.types.beta import (
    BetaAdvisorTool20260301Param,
    BetaToolParam,
    BetaToolUnionParam,
    BetaWebSearchTool20250305Param,
    BetaWebSearchTool20260209Param,
)

from aidial_adapter_anthropic.passthrough._helpers import (
    MessagesAPIEndpoint,
)
from aidial_adapter_anthropic.passthrough._middleware import (
    MessagesAPICloud,
    get_cloud_middlewares,
)
from tests.unit_tests.anthropic_mocks import (
    BASE_MESSAGES_REQUEST,
    MESSAGES_REQUEST,
    AnthropicAPIMock,
    AnthropicMocker,
    assert_unsupported_endpoint,
    read_fixture,
)

_AWS = MessagesAPICloud.AWS
_AZURE = MessagesAPICloud.AZURE
_GCP = MessagesAPICloud.GCP
_PLATFORM = MessagesAPICloud.PLATFORM
_EVERY_CLOUD = set(MessagesAPICloud)

_ADVISOR_FEATURE = "advisor-tool-2026-03-01"
# A flag no backend objects to: the control in every stripping assertion.
_UNIVERSAL_FEATURE = "token-efficient-tools-2025-02-19"

_WEB_SEARCH = BetaWebSearchTool20250305Param(
    type="web_search_20250305", name="web_search"
)
_WEB_SEARCH_NEXT = BetaWebSearchTool20260209Param(
    type="web_search_20260209", name="web_search"
)
_ADVISOR = BetaAdvisorTool20260301Param(
    type="advisor_20260301", name="advisor", model="claude-opus-5"
)
_CUSTOM = BetaToolParam(
    name="get_weather", input_schema={"type": "object", "properties": {}}
)


# A mocker asserts every route it registered was called, so each test mocks
# only the endpoint it exercises.
class _MessagesMock(AnthropicAPIMock):
    def on_block_messages(self, request) -> bytes:
        return read_fixture("messages_non_streaming_response.json")


class _CountTokensMock(AnthropicAPIMock):
    def on_count_tokens(self, request) -> bytes:
        return read_fixture("count_tokens_response.json")


class _BatchesMock(AnthropicAPIMock):
    def on_batches(self, request) -> bytes:
        return read_fixture("batches_response.json")


def _forwarded_betas(mocker: AnthropicMocker) -> list[str]:
    raw = mocker.router.calls.last.request.headers.get("anthropic-beta")
    return [f.strip() for f in raw.split(",")] if raw else []


def _forwarded_body(mocker: AnthropicMocker) -> dict:
    return json.loads(mocker.router.calls.last.request.content)


@pytest.mark.parametrize("endpoint", list(MessagesAPIEndpoint))
@pytest.mark.parametrize("body", [None, [1, 2, 3], "text", 42])
def test_a_non_object_body_is_left_to_the_upstream(body, endpoint):
    # A body that isn't a JSON object can't be rewritten, so it must reach the
    # upstream to be rejected there rather than blow up in a middleware.
    for middleware in get_cloud_middlewares(MessagesAPICloud.AWS):
        middleware.on_request({}, body, endpoint)


class TestBetaFeatures:
    """Every flag the package knows to be unimplemented by some cloud."""

    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        mocker.mock(_MessagesMock())

    @pytest.mark.parametrize(
        "feature, accepted_by",
        [
            (_UNIVERSAL_FEATURE, _EVERY_CLOUD),
            ("oauth-2025-04-20", _EVERY_CLOUD - {_AWS}),
            ("redact-thinking-2026-02-12", _EVERY_CLOUD - {_AWS}),
            ("claude-code-20250219", _EVERY_CLOUD - {_AWS}),
            ("advanced-tool-use-2025-11-20", _EVERY_CLOUD - {_AWS}),
            ("thinking-token-count-2026-05-13", {_PLATFORM, _AZURE}),
            ("prompt-caching-scope-2026-01-05", {_PLATFORM, _AZURE}),
            (_ADVISOR_FEATURE, {_PLATFORM}),
        ],
    )
    async def test_stripped_only_where_unsupported(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
        feature: str,
        accepted_by: set[MessagesAPICloud],
    ):
        await anthropic_client.beta.messages.create(
            **MESSAGES_REQUEST, betas=[feature, _UNIVERSAL_FEATURE]
        )

        forwarded = _forwarded_betas(mocker)
        assert (feature in forwarded) is (mocker.cloud in accepted_by)
        # Stripping one flag must leave the rest of the header intact.
        assert _UNIVERSAL_FEATURE in forwarded

    async def test_header_dropped_when_nothing_survives(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        # An empty anthropic-beta is rejected upstream, so the header must go
        # rather than be forwarded blank.
        await anthropic_client.beta.messages.create(
            **MESSAGES_REQUEST, betas=[_ADVISOR_FEATURE]
        )

        expected = [_ADVISOR_FEATURE] if mocker.cloud is _PLATFORM else []
        assert _forwarded_betas(mocker) == expected

    async def test_unknown_flag_is_relayed(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        # The package strips only what it knows a cloud rejects; anything else
        # is the upstream's business.
        await anthropic_client.beta.messages.create(
            **MESSAGES_REQUEST, betas=["some-future-flag-2030-01-01"]
        )

        assert _forwarded_betas(mocker) == ["some-future-flag-2030-01-01"]


class TestUnsupportedTools:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker: AnthropicMocker):
        mocker.mock(_MessagesMock())

    async def _create(
        self,
        client: anthropic.AsyncAnthropic,
        tools: list[BetaToolUnionParam],
        betas: list[str] | None = None,
    ) -> None:
        await client.beta.messages.create(
            **MESSAGES_REQUEST, tools=tools, betas=betas or []
        )

    async def test_web_search_dropped_on_aws(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        # Bedrock offers no web search at all, whichever dated version of the
        # tool the caller asks for.
        tools = [_WEB_SEARCH, _WEB_SEARCH_NEXT, _CUSTOM]
        await self._create(anthropic_client, tools)

        expected = [_CUSTOM] if mocker.cloud is _AWS else tools
        assert _forwarded_body(mocker)["tools"] == expected

    async def test_advisor_dropped_wherever_its_flag_is(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        # The advisor tool only runs alongside its beta flag, so it goes on
        # every cloud that rejects the flag.
        tools = [_ADVISOR, _CUSTOM]
        await self._create(anthropic_client, tools, betas=[_ADVISOR_FEATURE])

        expected = tools if mocker.cloud is _PLATFORM else [_CUSTOM]
        assert _forwarded_body(mocker)["tools"] == expected

    async def test_advisor_dropped_even_without_its_flag(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        # Whether the caller sent the flag changes nothing: on a cloud that
        # rejects it the tool could not have run either way.
        tools = [_ADVISOR, _CUSTOM]
        await self._create(anthropic_client, tools)

        expected = tools if mocker.cloud is _PLATFORM else [_CUSTOM]
        assert _forwarded_body(mocker)["tools"] == expected

    async def test_toolless_request_is_untouched(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        await anthropic_client.beta.messages.create(**MESSAGES_REQUEST)

        body = _forwarded_body(mocker)
        assert "tools" not in body
        assert body["messages"] == MESSAGES_REQUEST["messages"]


class TestEndpointsOtherThanMessages:
    async def test_count_tokens_tools_are_adapted(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        mocker.mock(_CountTokensMock())
        tools = [_WEB_SEARCH, _CUSTOM]

        async def _run() -> object:
            return await anthropic_client.beta.messages.count_tokens(
                **BASE_MESSAGES_REQUEST, tools=tools
            )

        if mocker.supports_count_tokens:
            await _run()
            expected = [_CUSTOM] if mocker.cloud is _AWS else tools
            assert _forwarded_body(mocker)["tools"] == expected
        else:
            await assert_unsupported_endpoint(_run)

    async def test_batched_tools_are_adapted(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
    ):
        # A batch nests one set of message params per queued request, so the
        # tools live a level down from where the other endpoints keep them.
        mocker.mock(_BatchesMock())
        tools = [_WEB_SEARCH, _CUSTOM]

        async def _run() -> object:
            return await anthropic_client.beta.messages.batches.create(
                requests=[
                    {
                        "custom_id": "req-1",
                        "params": {**MESSAGES_REQUEST, "tools": tools},  # type: ignore[typeddict-item]
                    }
                ]
            )

        if mocker.supports_batches:
            await _run()
            expected = [_CUSTOM] if mocker.cloud is _AWS else tools
            forwarded = _forwarded_body(mocker)["requests"][0]["params"]
            assert forwarded["tools"] == expected
        else:
            await assert_unsupported_endpoint(_run)

    @pytest.mark.parametrize("endpoint", ["count_tokens", "batches"])
    async def test_beta_features_are_stripped_too(
        self,
        mocker: AnthropicMocker,
        anthropic_client: anthropic.AsyncAnthropic,
        endpoint: str,
    ):
        # Every endpoint takes the anthropic-beta header, so every endpoint
        # needs the unsupported flags taken out of it.
        is_count_tokens = endpoint == "count_tokens"
        mocker.mock(_CountTokensMock() if is_count_tokens else _BatchesMock())

        async def _run() -> object:
            if is_count_tokens:
                return await anthropic_client.beta.messages.count_tokens(
                    **BASE_MESSAGES_REQUEST,
                    betas=[_ADVISOR_FEATURE, _UNIVERSAL_FEATURE],
                )
            return await anthropic_client.beta.messages.batches.create(
                requests=[{"custom_id": "req-1", "params": MESSAGES_REQUEST}],  # type: ignore[typeddict-item]
                betas=[_ADVISOR_FEATURE, _UNIVERSAL_FEATURE],
            )

        supported = (
            mocker.supports_count_tokens
            if is_count_tokens
            else mocker.supports_batches
        )
        if supported:
            await _run()
            forwarded = _forwarded_betas(mocker)
            assert (_ADVISOR_FEATURE in forwarded) is (
                mocker.cloud is _PLATFORM
            )
            assert _UNIVERSAL_FEATURE in forwarded
        else:
            await assert_unsupported_endpoint(_run)
