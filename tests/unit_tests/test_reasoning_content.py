from anthropic.types.beta import BetaThinkingBlock as ThinkingBlock

from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from tests.utils.consumer import invoke_non_streaming


async def test_thinking_populates_stage_and_reasoning_content(
    adapter: Adapter,
):
    consumer = await invoke_non_streaming(
        adapter,
        [
            ThinkingBlock(
                type="thinking", thinking="let me think", signature="sig"
            )
        ],
    )

    assert consumer.stages["Thinking"] == ["let me think"]
    assert consumer._choice.reasoning_content == ["let me think"]
