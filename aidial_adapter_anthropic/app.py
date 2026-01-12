from aidial_sdk import DIALApp

from aidial_adapter_anthropic.chat_completion import BedrockChatCompletion

app = DIALApp(
    description="Anthropic API adapter for DIAL API",
    add_healthcheck=True,
)

app.add_chat_completion("{deployment_id}", BedrockChatCompletion())
