from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .datastore import DecodedImage, LocalDataStoreV2
from .light_camera import LightCamera
from .resolution import compute_target_size
from .scene_index import FrameRecord, SceneManifest
from .specs import CameraParams
from .transforms import resize_alpha_tensor, resize_depth_tensor, resize_mask_tensor, resize_rgb_tensor


@dataclass(frozen=True)
class PreparedFrame:
    record: FrameRecord
    target_size: tuple[int, int]
    scaled_params: CameraParams
    camera: LightCamera
    image: torch.Tensor
    alpha: torch.Tensor | None
    assets: dict[str, torch.Tensor]
    global_assets: dict[str, torch.Tensor]

    def require_asset(self, name: str) -> torch.Tensor:
        asset = self.assets.get(name)
        if asset is None:
            raise KeyError(f"Prepared frame `{self.record.frame_id}` is missing required asset `{name}`")
        return asset

    def require_global_asset(self, name: str) -> torch.Tensor:
        asset = self.global_assets.get(name)
        if asset is None:
            raise KeyError(f"Prepared frame is missing required global asset `{name}`")
        return asset


class FrameTransformPipeline:
    def __init__(self, datastore: LocalDataStoreV2 | None = None):
        self.datastore = datastore or LocalDataStoreV2()

    def prepare(
        self,
        manifest: SceneManifest,
        record: FrameRecord,
        *,
        resolution: int,
        resolution_scale: float,
        required_assets: Iterable[str],
        required_global_assets: Iterable[str] = (),
    ) -> PreparedFrame:
        target_w, target_h = compute_target_size(
            record.params.width,
            record.params.height,
            resolution,
            resolution_scale,
        )
        target_size = (target_w, target_h)
        scaled_params = record.params.scaled_to(target_w, target_h)
        camera = LightCamera.from_params(scaled_params, scene_transform=manifest.scene_transform)

        decoded_image = self.datastore.load_image(record.require_asset("image"))
        image = resize_rgb_tensor(decoded_image.rgb, target_size)
        alpha = resize_alpha_tensor(decoded_image.alpha, target_size)

        assets: dict[str, torch.Tensor] = {}
        for asset_name in required_assets:
            if asset_name == "image":
                continue
            asset_ref = record.require_asset(asset_name)
            if asset_name == "masks":
                assets[asset_name] = resize_mask_tensor(self.datastore.load_masks(asset_ref), target_size)
            elif asset_name == "depth":
                assets[asset_name] = resize_depth_tensor(self.datastore.load_depth(asset_ref), target_size)
            elif asset_name == "confidence":
                assets[asset_name] = resize_depth_tensor(self.datastore.load_confidence(asset_ref), target_size)
            elif asset_name == "labels":
                assets[asset_name] = self.datastore.load_labels(asset_ref).long()
            else:
                loaded = self.datastore.load_asset(asset_ref)
                if not isinstance(loaded, torch.Tensor):
                    raise TypeError(
                        f"Expected tensor asset for `{asset_name}`, got {type(loaded).__name__}"
                    )
                assets[asset_name] = loaded

        global_assets: dict[str, torch.Tensor] = {}
        for asset_name in required_global_assets:
            asset_ref = manifest.require_global_asset(asset_name)
            loaded = self.datastore.load_asset(asset_ref)
            if not isinstance(loaded, torch.Tensor):
                raise TypeError(
                    f"Expected tensor global asset for `{asset_name}`, got {type(loaded).__name__}"
                )
            global_assets[asset_name] = loaded.float()

        return PreparedFrame(
            record=record,
            target_size=target_size,
            scaled_params=scaled_params,
            camera=camera,
            image=image,
            alpha=alpha,
            assets=assets,
            global_assets=global_assets,
        )
