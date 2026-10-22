from dataclasses import dataclass

from aidial_adapter_anthropic.adapter._decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_anthropic.dial.cache_info import DialCacheInfo
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import AdapterRequest


def caching_decorator() -> ChatCompletionTransformer:
    return lambda adapter: CachingDecorator(adapter=adapter)


@dataclass
class CachingDecorator(ChatCompletionDecorator):
    async def chat(self, consumer: Consumer, request: AdapterRequest) -> None:
        if info := DialCacheInfo.create(request):
            consumer.get_response().set_cache_breakpoint(
                cache_breakpoint_path=info.breakpoint_path,
                cache_expire_at=info.expired_at,
            )
        await self.adapter.chat(consumer, request)
