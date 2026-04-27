import base64

import pytest
from aidial_sdk.chat_completion import (
    InputAudio,
    InputFile,
    MessageContentAudioPart,
    MessageContentFilePart,
    MessageContentPart,
)

from aidial_adapter_anthropic._utils.resource import Resource
from aidial_adapter_anthropic.adapter._errors import ValidationError
from aidial_adapter_anthropic.dial._attachments import (
    AttachmentProcessor,
    AttachmentProcessors,
)
from aidial_adapter_anthropic.dial._message import HumanRegularMessage


@pytest.fixture
def attachment_processors() -> AttachmentProcessors:
    return AttachmentProcessors(
        text_handler=lambda text: text,
        attachment_processors=[
            AttachmentProcessor(
                supported_types={
                    "audio/wav": {"wav"},
                    "audio/foo": {"foo"},
                    "application/pdf": {"pdf"},
                    "text/plain": {"txt"},
                },
                handler=lambda resource: resource,
            )
        ],
        file_storage=None,
    )


async def _process_part(
    attachment_processors: AttachmentProcessors, part: MessageContentPart
) -> Resource:
    message = HumanRegularMessage(content=[part])
    [resource] = (
        await attachment_processors.process_attachments(message)
    ).payload
    return resource


async def test_processes_file_content_part_with_data_url(
    attachment_processors: AttachmentProcessors,
):
    data_url = Resource(type="text/plain", data=b"file-content").to_data_url()
    resource = await _process_part(
        attachment_processors,
        MessageContentFilePart(type="file", file=InputFile(file_data=data_url)),
    )
    assert resource.type == "text/plain"
    assert resource.data == b"file-content"


async def test_processes_file_content_part_with_base64_data(
    attachment_processors: AttachmentProcessors,
):
    b64data = base64.b64encode(b"file-content").decode()
    resource = await _process_part(
        attachment_processors,
        MessageContentFilePart(type="file", file=InputFile(file_data=b64data)),
    )
    assert resource.type == "application/pdf"
    assert resource.data == b"file-content"


async def test_rejects_invalid_file_content_part(
    attachment_processors: AttachmentProcessors,
):
    error_message = "Invalid file content part: file_data must be a valid data URL or base64 string"
    with pytest.raises(ValidationError, match=error_message):
        await _process_part(
            attachment_processors,
            MessageContentFilePart(type="file", file=InputFile(file_data="?!")),
        )


@pytest.mark.parametrize("format", ["wav", "foo"], ids=["wav", "custom-format"])
async def test_processes_audio_content_part(
    attachment_processors: AttachmentProcessors, format: str
):
    resource = await _process_part(
        attachment_processors,
        MessageContentAudioPart(
            type="input_audio",
            input_audio=InputAudio(
                data=base64.b64encode(b"audio-data").decode(), format=format
            ),
        ),
    )
    assert resource.type == f"audio/{format}"
    assert resource.data == b"audio-data"
