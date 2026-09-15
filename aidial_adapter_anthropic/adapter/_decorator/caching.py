from dataclasses import dataclass

from aidial_adapter_anthropic.adapter._claude.caching import (
    get_cache_info,
)
from aidial_adapter_anthropic.adapter._decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import AdapterRequest


def caching_decorator() -> ChatCompletionTransformer:
    return lambda adapter: CachingDecorator(adapter=adapter)


@dataclass
class CachingDecorator(ChatCompletionDecorator):
    async def chat(self, consumer: Consumer, request: AdapterRequest) -> None:
        tools = request.tool_config.tools if request.tool_config else []
        if info := get_cache_info(
            request.cache_breakpoint, request.messages, tools
        ):
            consumer.get_response().set_cache_breakpoint(
                cache_breakpoint_path=info.breakpoint_path,
                cache_expire_at=info.expired_at,
            )
        await self.adapter.chat(consumer, request)
