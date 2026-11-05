"""A JSON request body that is cheap to forward.

Encoding a large prompt back to JSON costs more CPU than the rest of the
passthrough, so the body remembers where each prompt field lies in the text
it was parsed from, and sends the prompt fields nothing has replaced as that
very text.
"""

import functools
import itertools
import json
from dataclasses import dataclass
from typing import Any, Self

from anthropic._models import FinalRequestOptions

from aidial_adapter_anthropic.passthrough._middleware import AnthropicClient

# The fields that make up nearly all of a large request. Only they are sent as
# their source text: the clients may route on any other field, and the
# stand-in they see instead of the value is always truthy.
_PROMPT_FIELDS = ("messages", "system", "tools")


@dataclass(frozen=True)
class RequestBody:
    """
    A JSON request body, parsed like ``json.loads`` does. Content that isn't
    JSON gets a ``json_data`` of ``None`` and is sent as received.

    ``json_data`` may be changed before the body is sent, but a top-level field
    counts as changed only when it is replaced as a whole: one edited in place
    is sent as it was received.
    """

    json_data: Any
    _content: bytes
    _text: str

    # top level field mapped to the json value and its span in the original string.
    _fields: dict[str, tuple[Any, slice]]

    @classmethod
    def parse(cls, content: bytes) -> Self:
        try:
            text = content.decode(
                json.detect_encoding(content), "surrogatepass"
            )
            json_data, spans = _parse(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return cls(json_data=None, _content=content, _text="", _fields={})
        fields = {
            k: (json_data[k], span)
            for k, span in spans.items()
            if k in _PROMPT_FIELDS
        }
        # The text is only kept for the prompt fields to be sliced out of it.
        return cls(
            json_data=json_data,
            _content=content,
            _text=text if fields else "",
            _fields=fields,
        )

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
        but sees the unreplaced prompt fields only as opaque stand-ins, which
        are cheap to copy.
        """
        if not isinstance(self.json_data, dict):
            # Nothing for the client to adapt: the body goes as received.
            return FinalRequestOptions(
                method=method,
                url=url,
                content=self._content,
                headers=headers,
            )

        raw = {
            k: _RawJSON(self._text[span])
            for k, (parsed, span) in self._fields.items()
            if k in self.json_data and self.json_data[k] is parsed
        }
        options = FinalRequestOptions(
            method=method,
            url=url,
            json_data={**self.json_data, **raw},
            headers=headers,
        )
        options = await client._prepare_options(options)
        if isinstance(options.json_data, dict):
            # The client prepares the options once more on sending, which a
            # body given as content makes a no-op for the body.
            options.content = _dumps_with_raw(options.json_data)
            options.json_data = None
        return options


@dataclass(frozen=True)
class _RawJSON:
    """Source text of a JSON value, to be written out as is."""

    text: str

    def __deepcopy__(self, memo: dict) -> Self:
        # Immutable: the clients deep-copy the body they adapt.
        return self


_dumps = functools.partial(
    json.dumps, ensure_ascii=False, separators=(",", ":"), allow_nan=False
)


def _dumps_with_raw(body: dict[str, Any]) -> bytes:
    """The SDK's JSON encoding of a dict whose ``_RawJSON`` fields go verbatim."""
    return (
        "{"
        + ",".join(
            f"{_dumps(k)}:{v.text if isinstance(v, _RawJSON) else _dumps(v)}"
            for k, v in body.items()
        )
        + "}"
    ).encode()


# Parses a JSON value at an index of a string, with the scanner `json.loads` uses.
_raw_decode = json.JSONDecoder().raw_decode


# Well above the number of Messages API params:
# https://platform.claude.com/docs/en/api/python/messages/create
_MAX_FIELDS = 64


def _skip_ws(s: str, idx: int) -> int:
    # The regex `json` itself skips whitespace with: a Python loop is ~100x slower.
    return json.decoder.WHITESPACE.match(s, idx).end()  # type: ignore


def _parse(s: str) -> tuple[Any, dict[str, slice]]:
    """
    ``json.loads`` that also returns, for a JSON object, where each top-level
    field's value lies in ``s``. It costs no more than ``json.loads``: the
    fields are parsed by the same scanner, one at a time. An object with more
    than ``_MAX_FIELDS`` fields gets no spans.
    """
    idx = _skip_ws(s, 0)
    if not s.startswith("{", idx):
        return json.loads(s), {}

    obj: dict[str, Any] = {}
    spans: dict[str, slice] = {}
    idx = _skip_ws(s, idx + 1)
    if s.startswith("}", idx):
        idx += 1
    else:
        for parsed in itertools.count(1):
            if not s.startswith('"', idx):
                raise json.JSONDecodeError(
                    "Expecting property name enclosed in double quotes", s, idx
                )
            key, idx = _raw_decode(s, idx)
            idx = _skip_ws(s, idx)
            if not s.startswith(":", idx):
                raise json.JSONDecodeError("Expecting ':' delimiter", s, idx)
            start = _skip_ws(s, idx + 1)
            value, idx = _raw_decode(s, start)
            obj[key] = value
            spans[key] = slice(start, idx)
            idx = _skip_ws(s, idx)
            if s.startswith(",", idx):
                if parsed == _MAX_FIELDS:
                    # Not a Messages API body: a Python loop over many keys
                    # costs several times what json.loads does (repeated keys too).
                    return json.loads(s), {}
                idx = _skip_ws(s, idx + 1)
            elif s.startswith("}", idx):
                idx += 1
                break
            else:
                raise json.JSONDecodeError("Expecting ',' delimiter", s, idx)

    idx = _skip_ws(s, idx)
    if idx != len(s):
        raise json.JSONDecodeError("Extra data", s, idx)
    return obj, spans
