"""A JSON request body that is cheap to forward.

The clients adapt the body to their cloud by deep-copying it, and encode it
with ``json``: on a large prompt, both cost more CPU than the rest of the
passthrough. So the body is parsed and encoded with orjson, and the clients
see the prompt fields only as stand-ins, which are cheap to copy.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import orjson
from anthropic._models import FinalRequestOptions

from aidial_adapter_anthropic.passthrough._middleware import AnthropicClient

# The fields that make up nearly all of a large request. Only they are hidden
# behind stand-ins: the clients may route on any other field, and a stand-in
# is always truthy.
_PROMPT_FIELDS = ("messages", "system", "tools")


@dataclass(frozen=True)
class RequestBody:
    """
    A JSON request body, parsed like ``json.loads`` does. Content that isn't
    JSON gets a ``json_data`` of ``None`` and is sent as received.
    """

    json_data: Any
    _content: bytes
    _dumps: Callable[[Any], bytes]

    @classmethod
    def parse(cls, content: bytes) -> Self:
        try:
            json_data = orjson.loads(content)
            # orjson reads an integer beyond 64 bits as a float, losing digits.
            if not _has_huge_float(json_data):
                return cls(json_data, content, _orjson_dumps)
        except orjson.JSONDecodeError:
            # What json takes and orjson rejects: lone surrogates, NaN and
            # infinities, UTF-16 and -32, a BOM.
            pass
        try:
            json_data = json.loads(content)
        except ValueError:
            json_data = None
        return cls(json_data, content, _json_dumps)

    async def to_request_options(
        self,
        client: AnthropicClient,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
    ) -> FinalRequestOptions:
        """
        The options that send the body through the ``client``. It still adapts
        the body to its cloud (Vertex and Bedrock move "model" into the URL),
        but sees the prompt fields only as stand-ins.
        """
        if not isinstance(self.json_data, dict):
            # Nothing for the client to adapt: the body goes as received.
            return FinalRequestOptions(
                method=method,
                url=url,
                content=self._content,
                headers=headers,
            )

        stand_ins = {
            k: _Opaque(v)
            for k, v in self.json_data.items()
            if k in _PROMPT_FIELDS
        }
        options = FinalRequestOptions(
            method=method,
            url=url,
            json_data={**self.json_data, **stand_ins},
            headers=headers,
        )
        options = await client._prepare_options(options)
        if isinstance(options.json_data, dict):
            # The client prepares the options once more on sending, which a
            # body given as content makes a no-op for the body.
            options.content = self._dumps(options.json_data)
            options.json_data = None
        return options


class _Opaque:
    """
    A value the clients pass along by reference. Not a dataclass: orjson
    would encode one as an object of its fields.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __deepcopy__(self, memo: dict) -> Self:
        # The clients deep-copy the body they adapt, but never change a prompt.
        return self


def _unwrap(value: Any) -> Any:
    if isinstance(value, _Opaque):
        return value.value
    raise TypeError(f"Type is not JSON serializable: {type(value).__name__}")


def _orjson_dumps(body: Any) -> bytes:
    return orjson.dumps(body, default=_unwrap)


def _json_dumps(body: Any) -> bytes:
    # ASCII: a lone surrogate can only be sent escaped.
    return json.dumps(body, default=_unwrap, separators=(",", ":")).encode()


def _has_huge_float(value: Any) -> bool:
    """Whether a parsed JSON value holds a float beyond the 64-bit integers."""
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, float):
            if abs(v) >= 2.0**63:
                return True
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
    return False
