import os
import warnings

if os.getenv("PYDANTIC_V2", "0").lower() not in ("1", "true"):
    warnings.warn("PYDANTIC_V2 env variable is expected to be set to True")
