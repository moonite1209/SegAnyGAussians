from __future__ import annotations

"""Shared configuration models used across SAGA pipeline stages."""

from pydantic import Field, model_validator

from .base import StrictConfigModel, require_non_empty

__all__ = ["ModelConfig", "DatasetConfig", "PipeConfig"]


class ModelConfig(StrictConfigModel):
    """Shared 3D Gaussian model parameters."""

    sh_degree: int = Field(ge=0)
    instance_feature_dim: int = Field(gt=0)
    semantic_feature_dim: int = Field(gt=0)
    white_background: bool = Field(default=False)


class DatasetConfig(StrictConfigModel):
    """Shared dataset input parameters."""

    base_path: str
    resolution: int
    eval: bool = Field(default=False)
    allow_principle_point_shift: bool = Field(default=False)

    images_path: str
    sparse_path: str | None = None
    source_path: str | None = None
    depth_path: str | None = None

    llffhold: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def _validate_semantics(self) -> "DatasetConfig":
        require_non_empty(self.base_path, "dataset.base_path")
        require_non_empty(self.images_path, "dataset.images_path")
        if self.resolution == 0:
            raise ValueError("`dataset.resolution` must not be 0")
        return self


class PipeConfig(StrictConfigModel):
    """Shared rendering pipeline toggles."""

    convert_SHs_python: bool = Field(default=False)
    compute_cov3D_python: bool = Field(default=False)
    debug: bool = Field(default=False)
