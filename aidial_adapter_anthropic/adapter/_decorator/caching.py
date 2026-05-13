from dataclasses import dataclass

from aidial_sdk.chat_completion import Message

from aidial_adapter_anthropic.adapter._claude.caching import (
    get_response_headers_for_caching,
)
from aidial_adapter_anthropic.adapter._decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import ModelParameters


def caching_decorator() -> ChatCompletionTransformer:
    return lambda adapter: CachingDecorator(adapter=adapter)


@dataclass
class CachingDecorator(ChatCompletionDecorator):
    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: list[Message],
    ) -> None:
        tools = params.tool_config.tools if params.tool_config else None
        if headers := get_response_headers_for_caching(
            params.cache_breakpoint, messages, tools
        ):
            await consumer.set_response_headers(headers)
        await self.adapter.chat(consumer, params, messages)
