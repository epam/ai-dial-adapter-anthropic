import logging
from typing import List

import pydantic
from anthropic.types.beta import BetaContentBlock as ContentBlock
from anthropic.types.beta import BetaContentBlockParam as ContentBlockParam
from anthropic.types.beta.parsed_beta_message import (
    ParsedBetaContentBlock as ParsedContentBlock,
)
from pydantic import BaseModel

from aidial_adapter_anthropic.dial._message import (
    AIRegularMessage,
    AIToolCallMessage,
)

_log = logging.getLogger(__name__)


class MessageState(BaseModel):
    claude_message_content: List[ParsedContentBlock] | List[ContentBlock]

    def to_dict(self) -> dict:
        return self.dict(
            # FIXME: a hack to exclude the private __json_buf field
            exclude={"claude_message_content": {"__all__": {"__json_buf"}}},
            # Excluding `citations: null`, since they could not be even parsed
            # currently by the Bedrock.
            exclude_none=True,
        )


def get_message_content_from_state(
    idx: int, message: AIRegularMessage | AIToolCallMessage
) -> List[ContentBlockParam] | None:
    if (cc := message.custom_content) and (state_dict := cc.state):
        try:
            state = MessageState.parse_obj(state_dict)
            return [block.to_dict() for block in state.claude_message_content]  # type: ignore
        except pydantic.ValidationError as e:
            _log.error(
                f"Invalid state at the path 'messages[{idx}].custom_content.state': {e}"
            )

    return None
