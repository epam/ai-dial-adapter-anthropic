"""Per-cloud adaptation of an Anthropic API request before it is forwarded."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
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

from aidial_adapter_anthropic.passthrough._helpers import (
    BETA_HEADER,
    MessagesAPIEndpoint,
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

    is_unsupported: Callable[[str], bool]

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

    def _not_supported(self, tool: object) -> bool:
        return (
            isinstance(tool, dict)
            and isinstance(tool_type := tool.get("type"), str)
            and self.is_unsupported(tool_type)
        )

    def _strip(self, params: dict) -> None:
        tools = params.get("tools")
        if not isinstance(tools, list):
            return

        dropped = sorted({t["type"] for t in tools if self._not_supported(t)})
        if dropped:
            params["tools"] = [t for t in tools if not self._not_supported(t)]
            _log.debug(f"Dropped tools unsupported by the upstream: {dropped}")


def _tool_containers(
    body: Any, endpoint: MessagesAPIEndpoint
) -> Iterator[dict]:
    if not isinstance(body, dict):
        # A malformed body is the upstream's to reject, not ours to rewrite.
        return

    if endpoint is MessagesAPIEndpoint.BATCHES:
        requests = body.get("requests")
        for request in requests if isinstance(requests, list) else []:
            if isinstance(request, dict) and isinstance(
                params := request.get("params"), dict
            ):
                yield params
    else:
        yield body


_ADVISOR_TOOL_FEATURE = "advisor-tool-2026-03-01"

# The body counterpart of a beta flag: what else has to go when the flag does.
_BETA_FEATURE_COMPANIONS: dict[AnthropicBetaParam, MessagesMiddleware] = {
    # The advisor tool is only usable together with its flag, so a cloud that
    # doesn't take the flag can't run the tool either.
    _ADVISOR_TOOL_FEATURE: RemoveTools(lambda t: t == "advisor_20260301"),
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
# Matched by prefix because the versions are dated (web_search_20250305,
# web_search_20260209, ...) and a list of them here would go stale behind the
# next one.
_remove_web_search = RemoveTools(lambda t: t.startswith("web_search_"))

_CLOUD_MIDDLEWARES: dict[MessagesAPICloud, list[MessagesMiddleware]] = {
    MessagesAPICloud.AWS: [
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
    ],
    MessagesAPICloud.GCP: unsupported_beta_features(
        [
            "thinking-token-count-2026-05-13",
            "prompt-caching-scope-2026-01-05",
            _ADVISOR_TOOL_FEATURE,
        ]
    ),
    MessagesAPICloud.AZURE: unsupported_beta_features([_ADVISOR_TOOL_FEATURE]),
    MessagesAPICloud.PLATFORM: [],
}


def get_cloud_middlewares(cloud: MessagesAPICloud) -> list[MessagesMiddleware]:
    return _CLOUD_MIDDLEWARES[cloud]
