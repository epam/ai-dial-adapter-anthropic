from pydantic import BaseModel, ConfigDict


class ExtraForbidModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnyTypeModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
