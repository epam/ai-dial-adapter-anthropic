import aidial_adapter_anthropic.llm.model.claude.v3.adapter as claude_v3
from aidial_adapter_anthropic.bedrock import create_anthropic_client
from aidial_adapter_anthropic.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_anthropic.upstream_config import UpstreamConfig


async def get_bedrock_adapter(
    *, deployment: str, api_key: str, upstream_config: UpstreamConfig
) -> ChatCompletionAdapter:
    client = await create_anthropic_client(upstream_config)

    return await claude_v3.create_adapter(
        deployment,
        api_key,
        client,
        supports_thinking=True,
        supports_documents=True,
    )
