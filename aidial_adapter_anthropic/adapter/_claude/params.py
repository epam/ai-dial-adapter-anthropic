from typing import TypedDict

from anthropic import Omit
from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam as CacheControlEphemeralParam,
)
from anthropic.types.beta import BetaOutputConfigParam as OutputConfigParam
from anthropic.types.beta import BetaTextBlockParam as TextBlockParam
from anthropic.types.beta import BetaThinkingConfigParam as ThinkingConfigParam
from anthropic.types.beta import BetaToolChoiceParam as ToolChoice
from anthropic.types.beta import BetaToolParam as ToolParam
from anthropic.types.beta import (
    BetaWebSearchTool20250305Param,
    BetaWebSearchTool20260209Param,
)

WebSearchToolParam = (
    BetaWebSearchTool20250305Param | BetaWebSearchTool20260209Param
)


class ClaudeParameters(TypedDict):
    """
    Subset of parameters to Anthropic Messages API request:
    https://github.com/anthropics/anthropic-sdk-python/blob/v0.95.0/src/anthropic/resources/beta/messages/messages.py#L1505-L1536
    """

    max_tokens: int
    stop_sequences: list[str] | Omit
    system: str | list[TextBlockParam] | Omit
    temperature: float | Omit
    top_p: float | Omit
    tools: list[ToolParam | WebSearchToolParam] | Omit
    tool_choice: ToolChoice | Omit
    thinking: ThinkingConfigParam | Omit
    betas: list[AnthropicBetaParam] | Omit
    output_config: OutputConfigParam | Omit
    cache_control: CacheControlEphemeralParam | Omit
