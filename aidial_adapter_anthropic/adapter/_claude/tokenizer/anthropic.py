from dataclasses import dataclass
from typing import List

from aidial_sdk.exceptions import InternalServerError
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
from anthropic._resource import AsyncAPIResource
from anthropic.resources.beta import AsyncMessages as FirstPartyAsyncMessagesAPI
from anthropic.types.beta import BetaMessageParam as ClaudeMessageParam

from aidial_adapter_anthropic.adapter._claude.params import ClaudeParameters

AnthropicClient = (
    AsyncAnthropic
    | AsyncAnthropicBedrock
    | AsyncAnthropicVertex
    | AsyncAnthropicFoundry
)


# Beta AsyncMessages doesn't provide the 'count_tokens' method,
# so we enabled it via the adapter.
class _AsyncMessagesAdapter(AsyncAPIResource):
    count_tokens = FirstPartyAsyncMessagesAPI.count_tokens

    def __init__(self, resource: AsyncAPIResource):
        super().__init__(resource._client)


@dataclass
class AnthropicTokenizer:
    deployment: str
    client: AnthropicClient

    def tokenize_text(self, text: str) -> int:
        raise InternalServerError(
            "Tokenization of strings is not supported by Anthropic API"
        )

    async def tokenize(
        self, params: ClaudeParameters, messages: List[ClaudeMessageParam]
    ) -> int:
        return (
            await _AsyncMessagesAdapter(self.client.beta.messages).count_tokens(
                model=self.deployment,
                messages=messages,
                system=params["system"],
                thinking=params["thinking"],
                tools=params["tools"],
                tool_choice=params["tool_choice"],
                betas=params["betas"],
            )
        ).input_tokens
