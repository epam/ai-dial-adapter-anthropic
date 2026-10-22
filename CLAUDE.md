# Code conventions

Rules from code-owner reviews. Follow them in new code.

## Python

- No `assert` outside tests: `python -O` strips it. Use an `isinstance` check that
  raises `TypeError` with an explanatory message, or falls back to the default behavior.
- Annotate a method that returns its own class with `typing.Self`, not a quoted class name.
- Use the precise type a function relies on, not `Any`: a function that calls
  `.items()` and needs string keys takes `dict[str, Any]`.
- Prefer a public stdlib API over its internals when it does the same job:
  `json.JSONDecoder().raw_decode`, not `json.scanner.make_scanner`.

## Comments

- When a comment states a fact about an external API (limits, params, behavior),
  link the docs, e.g. https://platform.claude.com/docs/en/api/python/messages/create
