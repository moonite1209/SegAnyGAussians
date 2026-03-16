from .feature_gaussian_model import FeatureGaussianModel
from .gaussian_model import GaussianModel

__all__ = ["GaussianModel", "FeatureGaussianModel", "Scene", "FeatureScene"]


class Scene:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "`scene.Scene` has been retired. Use `saga_data.build_scene_index(...)` "
            "with `saga_data.RenderDataset` or `saga_data.FeatureDataset` instead."
        )


class FeatureScene:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "`scene.FeatureScene` has been retired. Use `saga_data.FeatureDataset` instead."
        )
