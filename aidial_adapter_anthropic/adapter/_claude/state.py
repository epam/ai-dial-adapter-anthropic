import logging

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
    claude_message_content: list[ParsedContentBlock] | list[ContentBlock]

    def to_dict(self) -> dict:
        return self.model_dump(
            # FIXME: a hack to exclude the private __json_buf field
            exclude={"claude_message_content": {"__all__": {"__json_buf"}}},
            # Excluding `citations: null`, since they could not be even parsed
            # currently by the Bedrock.
            exclude_none=True,
        )


def get_message_content_from_state(
    idx: int, message: AIRegularMessage | AIToolCallMessage
) -> list[ContentBlockParam] | None:
    # The state may have been written by another adapter,
    # in which case the Claude-specific field is simply absent.
    if (
        (cc := message.custom_content)
        and (state_dict := cc.state)
        and "claude_message_content" in state_dict
    ):
        try:
            state = MessageState.model_validate(state_dict)
            return [block.to_dict() for block in state.claude_message_content]  # type: ignore
        except pydantic.ValidationError as e:
            _log.error(
                f"Invalid state at the path 'messages[{idx}].custom_content.state': {e}"
            )

    return None
