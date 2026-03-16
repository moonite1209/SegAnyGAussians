import numpy as np
import torch

from .scene_index import SceneTransform
from .specs import CameraParams


class LightCamera:
    """
    Minimal camera class for rendering. Mirrors gaussian_renderer expectations.
    """

    def __init__(
        self,
        params: CameraParams,
        trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
        znear: float = 0.01,
        zfar: float = 100.0,
    ):
        self._params = params
        self.trans = trans
        self.scale = scale
        self.znear = znear
        self.zfar = zfar

        self._transform_cache = {}
        self._init_transforms()

    @classmethod
    def from_params(
        cls,
        params: CameraParams,
        scene_transform: SceneTransform | None = None,
        znear: float = 0.01,
        zfar: float = 100.0,
    ) -> "LightCamera":
        translate = scene_transform.translate if scene_transform is not None else np.zeros(3, dtype=np.float32)
        return cls(params=params, trans=translate, znear=znear, zfar=zfar)

    def _init_transforms(self):
        from utils.graphics_utils import getProjectionMatrixShift, getWorld2View2

        wv_transform = getWorld2View2(self._params.R, self._params.T, self.trans, self.scale)
        self._world_view_transform_np = wv_transform
        proj_matrix = getProjectionMatrixShift(
            self.znear,
            self.zfar,
            self._params.fx,
            self._params.fy,
            self._params.cx,
            self._params.cy,
            self._params.width,
            self._params.height,
            self.FoVx,
            self.FoVy,
        )
        self._projection_matrix_np = proj_matrix

    @property
    def uid(self) -> int:
        return self._params.uid

    @property
    def image_width(self) -> int:
        return self._params.width

    @property
    def image_height(self) -> int:
        return self._params.height

    @property
    def FoVx(self) -> float:
        return self._params.FoVx

    @property
    def FoVy(self) -> float:
        return self._params.FoVy

    @property
    def cx(self) -> float:
        return self._params.cx

    @property
    def cy(self) -> float:
        return self._params.cy

    @property
    def fx(self) -> float:
        return self._params.fx

    @property
    def fy(self) -> float:
        return self._params.fy

    @property
    def world_view_transform(self) -> torch.Tensor:
        if "world_view_transform" not in self._transform_cache:
            matrix = torch.tensor(self._world_view_transform_np).transpose(0, 1)
            self._transform_cache["world_view_transform"] = matrix
        return self._transform_cache["world_view_transform"]

    @property
    def projection_matrix(self) -> torch.Tensor:
        if "projection_matrix" not in self._transform_cache:
            matrix = torch.tensor(self._projection_matrix_np).transpose(0, 1)
            self._transform_cache["projection_matrix"] = matrix
        return self._transform_cache["projection_matrix"]

    @property
    def full_proj_transform(self) -> torch.Tensor:
        return (self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0))
        ).squeeze(0)

    @property
    def camera_center(self) -> torch.Tensor:
        return self.world_view_transform.inverse()[3, :3]

    def to(self, device: str) -> "LightCamera":
        for key in self._transform_cache:
            self._transform_cache[key] = self._transform_cache[key].to(device)
        return self
