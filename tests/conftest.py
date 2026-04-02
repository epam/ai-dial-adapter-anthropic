import anthropic
import pytest

from aidial_adapter_anthropic.adapter._claude.adapter import Adapter
from aidial_adapter_anthropic.adapter._claude.tokenizer import (
    ApproximateTokenizer,
)


@pytest.fixture
def adapter() -> Adapter:
    return Adapter(
        deployment="test-deployment",
        storage=None,
        client=anthropic.AsyncAnthropic(),
        tokenizer=ApproximateTokenizer(),
        default_max_tokens=1024,
        supports_thinking=True,
        supports_documents=True,
    )
