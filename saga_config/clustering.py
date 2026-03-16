from __future__ import annotations

from pydantic import Field, model_validator

from .base import StrictConfigModel, require_non_empty


class ClusteringConfig(StrictConfigModel):
    classes: list[str]
    selected_classes: list[str]
    segment_masks_dir: str
    segment_labels_dir: str
    segment_label_features_path: str
    feature_point_cloud_path: str
    json_path: str

    clean: bool = False
    quiet: bool = False
    k: int = Field(gt=0)

    instance_feature_ratio: float = Field(ge=0, le=1)
    semantic_feature_ratio: float = Field(ge=0, le=1)
    xyz_feature_ratio: float = Field(ge=0, le=1)

    instance_threshold: float = Field(ge=0, le=1)
    background_threshold: float = Field(ge=0, le=1)
    scale_threshold: float = Field(ge=0, le=1)
    opacity_threshold: float = Field(default=0.01, ge=0, le=1)

    sample_num: int = -1

    use_sor: bool = True
    sor_nb_neighbors: int = Field(default=50, gt=0)
    sor_std_ratio: float = Field(default=0.05, ge=0)

    @model_validator(mode="after")
    def _validate_semantics(self) -> "ClusteringConfig":
        if not self.classes:
            raise ValueError("`clustering.classes` must not be empty")
        require_non_empty(self.segment_masks_dir, "clustering.segment_masks_dir")
        require_non_empty(self.segment_labels_dir, "clustering.segment_labels_dir")
        require_non_empty(self.segment_label_features_path, "clustering.segment_label_features_path")
        require_non_empty(self.feature_point_cloud_path, "clustering.feature_point_cloud_path")
        require_non_empty(self.json_path, "clustering.json_path")

        selected = set(self.selected_classes)
        known = set(self.classes)
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"`clustering.selected_classes` contains unknown classes: {unknown}")

        total_ratio = self.instance_feature_ratio + self.semantic_feature_ratio + self.xyz_feature_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                "`clustering.instance_feature_ratio + clustering.semantic_feature_ratio + "
                "clustering.xyz_feature_ratio` must sum to 1.0"
            )
        return self
