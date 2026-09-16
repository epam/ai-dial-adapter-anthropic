from abc import ABC, abstractmethod

from pydantic import BaseModel

from aidial_adapter_anthropic._utils.list import ListProjection
from aidial_adapter_anthropic.adapter._errors import ValidationError
from aidial_adapter_anthropic.adapter._truncate_prompt import DiscardedMessages
from aidial_adapter_anthropic.dial._message import AdapterMessage, SystemMessage
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import AdapterRequest


class ChatCompletionAdapter(ABC):
    @abstractmethod
    async def chat(self, consumer: Consumer, request: AdapterRequest) -> None:
        pass

    async def configuration(self) -> type[BaseModel]:
        raise NotImplementedError

    async def count_prompt_tokens(self, request: AdapterRequest) -> int:
        raise NotImplementedError

    async def count_completion_tokens(self, string: str) -> int:
        raise NotImplementedError

    async def compute_discarded_messages(
        self, request: AdapterRequest
    ) -> DiscardedMessages | None:
        """
        The method truncates the list of messages to fit
        into the token limit set in `request.max_prompt_tokens`.

        If the limit isn't provided, then it returns None.
        Otherwise, returns the indices of _discarded_ messages which should be
        removed from the list to make the rest fit into the token limit.
        """
        raise NotImplementedError


def default_preprocess_messages(
    messages: ListProjection[AdapterMessage],
) -> ListProjection[AdapterMessage]:
    def _is_empty_system_message(msg: AdapterMessage) -> bool:
        return isinstance(msg, SystemMessage) and msg.text_content.strip() == ""

    ret: list[tuple[AdapterMessage, set[int]]] = []
    idx: set[int] = set()

    # A dropped message is attributed to the message that follows it.
    for msg, indices in messages.lst:
        idx |= indices
        if _is_empty_system_message(msg):
            continue
        ret.append((msg, idx))
        idx = set()

    if len(ret) == 0:
        raise ValidationError("List of messages must not be empty")

    return ListProjection(ret)
