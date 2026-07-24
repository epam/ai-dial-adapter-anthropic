from anthropic.types.beta import BetaUsage
from anthropic.types.beta.beta_output_tokens_details import (
    BetaOutputTokensDetails,
)

from aidial_adapter_anthropic.adapter._claude.converters import to_dial_usage


def test_to_dial_usage_maps_cache_and_reasoning_tokens():
    usage = BetaUsage(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=30,
        cache_read_input_tokens=20,
        output_tokens_details=BetaOutputTokensDetails(thinking_tokens=15),
    )

    result = to_dial_usage(usage)

    # Anthropic input_tokens excludes cache tokens; prompt_tokens is inclusive.
    assert result.prompt_tokens == 150
    assert result.completion_tokens == 50
    assert result.cache_read_input_tokens == 20
    assert result.cache_write_input_tokens == 30
    assert result.reasoning_tokens == 15


def test_to_dial_usage_defaults_when_details_absent():
    usage = BetaUsage(input_tokens=10, output_tokens=5)

    result = to_dial_usage(usage)

    assert result.prompt_tokens == 10
    assert result.cache_read_input_tokens == 0
    assert result.cache_write_input_tokens == 0
    assert result.reasoning_tokens == 0
