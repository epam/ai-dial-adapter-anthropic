import asyncio
from typing import List

from aidial_sdk.chat_completion import Message

from aidial_adapter_anthropic.adapter._decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import ModelParameters


def replicator_decorator() -> ChatCompletionTransformer:
    return lambda adapter: ReplicatorDecorator(adapter=adapter)


class ReplicatorDecorator(ChatCompletionDecorator):
    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:
        params1 = params.copy()
        params1.n = 1

        async def _chat(root_consumer: Consumer):
            with root_consumer.fork() as consumer:
                await self.adapter.chat(consumer, params1, messages)

        await asyncio.gather(*(_chat(consumer) for _ in range(params.n)))
