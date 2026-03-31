from typing import Protocol, runtime_checkable

from anthropic.types.beta import BetaMessageParam as ClaudeMessage

from aidial_adapter_anthropic.adapter._claude.params import ClaudeParameters
from aidial_adapter_anthropic.dial._attachments import WithResources


@runtime_checkable
class ClaudeTokenizer(Protocol):
    def tokenize_text(self, text: str) -> int: ...

    async def tokenize(
        self, params: ClaudeParameters, messages: list[ClaudeMessage]
    ) -> int: ...


def create_tokenizer(tokenizer: ClaudeTokenizer, params: ClaudeParameters):
    async def _tokenize(
        messages: list[tuple[WithResources[ClaudeMessage], set[int]]],
    ) -> int:
        return await tokenizer.tokenize(
            params, [msg[0].payload for msg in messages]
        )

    return _tokenize
