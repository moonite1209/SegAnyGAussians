from __future__ import annotations

from pydantic import Field, model_validator

from .base import StrictConfigModel, require_non_empty


class GuiConfig(StrictConfigModel):
    feature_point_cloud_path: str
    json_path: str
    quiet: bool = False
    window_height: int = Field(gt=0)
    window_width: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_semantics(self) -> "GuiConfig":
        require_non_empty(self.feature_point_cloud_path, "gui.feature_point_cloud_path")
        require_non_empty(self.json_path, "gui.json_path")
        return self
