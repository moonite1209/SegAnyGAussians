from __future__ import annotations

from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel

try:
    from pydantic import ConfigDict
except ImportError:  # pydantic v1
    ConfigDict = None


ModelT = TypeVar("ModelT", bound="StrictConfigModel")


class StrictConfigModel(BaseModel):
    """Pydantic base model with strict unknown-field rejection."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
    else:  # pydantic v1 fallback
        class Config:
            extra = "forbid"
            validate_assignment = True

    @classmethod
    def validate_model(cls: Type[ModelT], data: Dict[str, Any]) -> ModelT:
        if hasattr(cls, "model_validate"):
            return cls.model_validate(data)  # type: ignore[attr-defined]
        return cls.parse_obj(data)  # type: ignore[attr-defined]

    def dump_model(self, *, by_alias: bool = False, exclude_none: bool = False) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump(by_alias=by_alias, exclude_none=exclude_none)  # type: ignore[attr-defined]
        return self.dict(by_alias=by_alias, exclude_none=exclude_none)  # type: ignore[attr-defined]


def require_non_empty(value: str | None, field_name: str) -> None:
    if value is None or not str(value).strip():
        raise ValueError(f"`{field_name}` must be a non-empty string")
