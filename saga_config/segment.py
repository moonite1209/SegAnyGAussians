from __future__ import annotations

from pydantic import Field, model_validator

from .base import StrictConfigModel, require_non_empty


class GroundingDINODetectorConfig(StrictConfigModel):
    target: str = Field(alias="_target_")
    model_id: str
    device: str = "cuda"
    box_threshold: float = Field(default=0.35, ge=0, le=1)
    text_threshold: float = Field(default=0.35, ge=0, le=1)

    @model_validator(mode="after")
    def _validate_semantics(self) -> "GroundingDINODetectorConfig":
        require_non_empty(self.target, "segment.ovsegmenter.detector._target_")
        require_non_empty(self.model_id, "segment.ovsegmenter.detector.model_id")
        require_non_empty(self.device, "segment.ovsegmenter.detector.device")
        return self


class SAMSegmenterConfig(StrictConfigModel):
    target: str = Field(alias="_target_")
    model_id: str
    device: str = "cuda"

    @model_validator(mode="after")
    def _validate_semantics(self) -> "SAMSegmenterConfig":
        require_non_empty(self.target, "segment.ovsegmenter.segmenter._target_")
        require_non_empty(self.model_id, "segment.ovsegmenter.segmenter.model_id")
        require_non_empty(self.device, "segment.ovsegmenter.segmenter.device")
        return self


class OVDetectSegmenterConfig(StrictConfigModel):
    target: str = Field(alias="_target_")
    detector: GroundingDINODetectorConfig
    segmenter: SAMSegmenterConfig

    @model_validator(mode="after")
    def _validate_semantics(self) -> "OVDetectSegmenterConfig":
        require_non_empty(self.target, "segment.ovsegmenter._target_")
        return self


class SegmentConfig(StrictConfigModel):
    classes: list[str]
    ovsegmenter: OVDetectSegmenterConfig
    output_dir: str
    masks_dir: str
    labels_dir: str
    label_features_path: str
    rgb_masks_dir: str | None = None

    @model_validator(mode="after")
    def _validate_semantics(self) -> "SegmentConfig":
        if not self.classes:
            raise ValueError("`segment.classes` must not be empty")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("`segment.classes` must not contain duplicate entries")
        require_non_empty(self.output_dir, "segment.output_dir")
        require_non_empty(self.masks_dir, "segment.masks_dir")
        require_non_empty(self.labels_dir, "segment.labels_dir")
        require_non_empty(self.label_features_path, "segment.label_features_path")
        return self
