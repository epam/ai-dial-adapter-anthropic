"""Per-cloud adaptation of an Anthropic API request before it is forwarded."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, assert_never

from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
from anthropic.types import AnthropicBetaParam
from anthropic.types.beta import BetaToolUnionParam
from anthropic.types.beta.message_count_tokens_params import (
    MessageCountTokensParams,
)
from anthropic.types.beta.message_create_params import MessageCreateParamsBase

from aidial_adapter_anthropic.passthrough._helpers import (
    BETA_HEADER,
    MessagesAPIEndpoint,
    typecast_request_body,
)

_log = logging.getLogger(__name__)

AnthropicClient = (
    AsyncAnthropic
    | AsyncAnthropicBedrock
    | AsyncAnthropicBedrockMantle
    | AsyncAnthropicVertex
    | AsyncAnthropicFoundry
)


class MessagesAPICloud(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    PLATFORM = "platform"


def get_cloud(client: AnthropicClient) -> MessagesAPICloud:
    match client:
        case AsyncAnthropicBedrock() | AsyncAnthropicBedrockMantle():
            return MessagesAPICloud.AWS
        case AsyncAnthropicVertex():
            return MessagesAPICloud.GCP
        case AsyncAnthropicFoundry():
            return MessagesAPICloud.AZURE
        case AsyncAnthropic():
            return MessagesAPICloud.PLATFORM
        case _:
            assert_never(client)


_ToolContainer = MessageCreateParamsBase | MessageCountTokensParams


class MessagesMiddleware(ABC):
    @abstractmethod
    def on_request(
        self,
        headers: dict[str, str],
        body: Any | None,
        endpoint: MessagesAPIEndpoint,
    ) -> None: ...


@dataclass(frozen=True)
class RemoveBetaFeature(MessagesMiddleware):
    """
    Drops one flag from the `anthropic-beta` header.
    Applies to every Message API endpoint: they all accept the header.
    """

    feature: AnthropicBetaParam

    def on_request(
        self,
        headers: dict[str, str],
        body: Any | None,
        endpoint: MessagesAPIEndpoint,
    ) -> None:
        raw = headers.get(BETA_HEADER)
        if raw is None:
            return

        kept = [
            feature
            for f in raw.split(",")
            if (feature := f.strip()) and feature != self.feature
        ]
        if kept:
            headers[BETA_HEADER] = ",".join(kept)
        else:
            # An empty anthropic-beta is rejected upstream.
            del headers[BETA_HEADER]


@dataclass(frozen=True)
class RemoveTools(MessagesMiddleware):
    """
    Drops the server tools the upstream has no implementation of.
    The upstream rejects the whole request with a 400 over a tool type it
    doesn't know, so a tool it cannot run is better dropped than forwarded.
    """

    unsupported_tools: list[str]

    def on_request(
        self,
        headers: dict[str, str],
        body: Any | None,
        endpoint: MessagesAPIEndpoint,
    ) -> None:
        if body is None:
            return

        for params in _tool_containers(body, endpoint):
            self._strip(params)

    def _strip(self, params: _ToolContainer) -> None:
        if (tools := params.get("tools")) is None:
            return None

        kept: list[BetaToolUnionParam] = []
        dropped: set[str] = set()
        for tool in tools:
            if (name := tool.get("name")) in self.unsupported_tools:
                dropped.add(name)
            else:
                kept.append(tool)

        if dropped:
            params["tools"] = kept
            _log.debug(
                f"Dropped tools unsupported by the upstream: {sorted(dropped)}"
            )


def _tool_containers(
    body: Any, endpoint: MessagesAPIEndpoint
) -> Iterator[_ToolContainer]:
    match endpoint:
        case MessagesAPIEndpoint.POST_MESSAGES:
            if typecast_request_body(body, endpoint):
                yield body
        case MessagesAPIEndpoint.POST_COUNT_TOKENS:
            if typecast_request_body(body, endpoint):
                yield body
        case MessagesAPIEndpoint.POST_BATCHES:
            if typecast_request_body(body, endpoint):
                for request in body["requests"]:
                    yield request["params"]
        case _:
            assert_never(endpoint)


_ADVISOR_TOOL_FEATURE = "advisor-tool-2026-03-01"

# The body counterpart of a beta flag: what else has to go when the flag does.
_BETA_FEATURE_COMPANIONS: dict[AnthropicBetaParam, MessagesMiddleware] = {
    # The advisor tool is only usable together with its flag, so a cloud that
    # doesn't take the flag can't run the tool either.
    _ADVISOR_TOOL_FEATURE: RemoveTools(["advisor"]),
}


def unsupported_beta_features(
    features: Iterable[AnthropicBetaParam],
) -> list[MessagesMiddleware]:
    ret: list[MessagesMiddleware] = []
    for feature in features:
        ret.append(RemoveBetaFeature(feature))
        if (companion := _BETA_FEATURE_COMPANIONS.get(feature)) is not None:
            ret.append(companion)
    return ret


# No version of the web search tool is offered by Bedrock:
# https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
# -- "Web search is not available on Amazon Bedrock."
_remove_web_search = RemoveTools(["web_search"])


def apply_middlewares(
    client: AnthropicClient,
    headers: dict[str, str],
    body: Any | None,
    endpoint: MessagesAPIEndpoint,
) -> None:
    for middleware in get_cloud_middlewares(get_cloud(client)):
        try:
            middleware.on_request(headers, body, endpoint)
        except Exception:
            _log.exception("Failed to apply middleware to the request.")


def get_cloud_middlewares(cloud: MessagesAPICloud) -> list[MessagesMiddleware]:
    match cloud:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html
        case MessagesAPICloud.AWS:
            return [
                *unsupported_beta_features(
                    [
                        "oauth-2025-04-20",
                        "redact-thinking-2026-02-12",
                        "thinking-token-count-2026-05-13",
                        "prompt-caching-scope-2026-01-05",
                        "claude-code-20250219",
                        "advanced-tool-use-2025-11-20",
                        _ADVISOR_TOOL_FEATURE,
                    ]
                ),
                _remove_web_search,
            ]
        # https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai#features-not-supported
        case MessagesAPICloud.GCP:
            return unsupported_beta_features(
                [
                    "thinking-token-count-2026-05-13",
                    "prompt-caching-scope-2026-01-05",
                    _ADVISOR_TOOL_FEATURE,
                ]
            )
        # https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry#claude-features-not-supported-for-claude-in-microsoft-foundry
        case MessagesAPICloud.AZURE:
            return unsupported_beta_features([_ADVISOR_TOOL_FEATURE])
        # Nothing to strip; the per-cloud matrix the cases above follow:
        # https://platform.claude.com/docs/en/build-with-claude/overview
        case MessagesAPICloud.PLATFORM:
            return []
        case _:
            assert_never(cloud)
