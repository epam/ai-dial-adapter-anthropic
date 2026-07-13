from typing import Any, Literal

from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import BetaThinkingConfigParam as ThinkingConfigParam
from pydantic import BaseModel, ConfigDict, Field

from aidial_adapter_anthropic.adapter._claude.params import WebSearchToolParam

ClaudeEffort = Literal["low", "medium", "high", "xhigh", "max"]


class ExtraForbidModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaudeConfiguration(ExtraForbidModel):
    betas: list[AnthropicBetaParam] | None = Field(
        default=None,
        description="List of beta features to enable. Make sure to check if the given feature is supported by the Claude deployment you are using.",
    )
    enable_citations: bool = False
    web_search: WebSearchToolParam | None = Field(
        default=None,
        description=(
            "Anthropic web search server-tool definition. "
            "See https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool"
        ),
    )


class ClaudeConfigurationWithThinking(ClaudeConfiguration):
    thinking: ThinkingConfigParam | dict[str, Any] | None = None
    effort: ClaudeEffort | str | None = None


Configuration = ClaudeConfiguration | ClaudeConfigurationWithThinking
