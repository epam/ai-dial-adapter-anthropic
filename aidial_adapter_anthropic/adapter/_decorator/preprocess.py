from collections.abc import Callable
from dataclasses import dataclass, replace

from aidial_adapter_anthropic._utils.list import ListProjection
from aidial_adapter_anthropic.adapter._decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_anthropic.adapter._truncate_prompt import DiscardedMessages
from aidial_adapter_anthropic.dial._message import AdapterMessage
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import AdapterRequest

OnMessages = Callable[[list[AdapterMessage]], ListProjection[AdapterMessage]]


def preprocess_messages_decorator(
    on_messages: OnMessages,
) -> ChatCompletionTransformer:
    return lambda adapter: PreprocessMessagesDecorator(
        on_messages=on_messages, adapter=adapter
    )


@dataclass
class PreprocessMessagesDecorator(ChatCompletionDecorator):
    on_messages: OnMessages

    def _preprocess(
        self, request: AdapterRequest
    ) -> tuple[AdapterRequest, ListProjection[AdapterMessage]]:
        messages = self.on_messages(request.messages)
        return replace(request, messages=messages.raw_list), messages

    async def chat(self, consumer: Consumer, request: AdapterRequest) -> None:
        new_request, new_messages = self._preprocess(request)
        await self.adapter.chat(consumer, new_request)
        if (
            discarded_messages := await consumer.get_discarded_messages()
        ) is not None:
            discarded_messages = list(
                new_messages.to_original_indices(discarded_messages)
            )
            await consumer.set_discarded_messages(discarded_messages)

    async def count_prompt_tokens(self, request: AdapterRequest) -> int:
        new_request, _ = self._preprocess(request)
        return await self.adapter.count_prompt_tokens(new_request)

    async def compute_discarded_messages(
        self, request: AdapterRequest
    ) -> DiscardedMessages | None:
        new_request, new_messages = self._preprocess(request)
        discarded_messages = await self.adapter.compute_discarded_messages(
            new_request
        )

        if discarded_messages is not None:
            discarded_messages = list(
                new_messages.to_original_indices(discarded_messages)
            )

        return discarded_messages
