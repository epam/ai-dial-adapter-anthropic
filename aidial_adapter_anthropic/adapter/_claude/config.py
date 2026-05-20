from typing import Any

from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import BetaThinkingConfigParam as ThinkingConfigParam
from pydantic import BaseModel, ConfigDict, Field


class ExtraForbidModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaudeConfiguration(ExtraForbidModel):
    betas: list[AnthropicBetaParam] | None = Field(
        default=None,
        description="List of beta features to enable. Make sure to check if the given feature is supported by the Claude deployment you are using.",
    )
    enable_citations: bool = False


class ClaudeConfigurationWithThinking(ClaudeConfiguration):
    # NOTE: once migrated to Pydantic v2 we can use TypeAdapter over
    # the anthropic's ThinkingConfigParam class directly.
    thinking: ThinkingConfigParam | dict[str, Any] | None = None


Configuration = ClaudeConfiguration | ClaudeConfigurationWithThinking
