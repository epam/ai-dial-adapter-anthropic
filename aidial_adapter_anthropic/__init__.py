import os

if os.getenv("PYDANTIC_V2", "0").lower() not in ("1", "true"):
    raise ValueError("PYDANTIC_V2 env variable is expected to be set to True")
