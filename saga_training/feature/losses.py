from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InstanceLossResult:
    loss: torch.Tensor
    valid_masks: int
    used_background: bool


@dataclass(frozen=True)
class SemanticLossResult:
    loss: torch.Tensor
    valid_masks: int


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _normalized_mask_mean(feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.normalize(feature_map[:, mask].mean(dim=1), dim=0)


def mask_prototype_info_nce_loss(
    masks: torch.Tensor,
    rendered_features: torch.Tensor,
    temperature: float,
) -> InstanceLossResult:
    if masks.ndim != 3:
        raise ValueError("`masks` must have shape [N, H, W]")
    if rendered_features.ndim != 3:
        raise ValueError("`rendered_features` must have shape [C, H, W]")

    valid_indices: list[int] = []
    prototypes: list[torch.Tensor] = []
    for idx, mask in enumerate(masks):
        if bool(mask.any()):
            valid_indices.append(idx)
            prototypes.append(_normalized_mask_mean(rendered_features, mask))

    background_mask = ~masks.any(dim=0)
    background_prototype = None
    if bool(background_mask.any()):
        background_prototype = _normalized_mask_mean(rendered_features, background_mask)

    if not valid_indices:
        return InstanceLossResult(
            loss=_zero_loss(rendered_features),
            valid_masks=0,
            used_background=background_prototype is not None,
        )

    mask_losses: list[torch.Tensor] = []
    detached_prototypes = [prototype.detach() for prototype in prototypes]
    detached_background = None if background_prototype is None else background_prototype.detach()

    for index_position, mask_index in enumerate(valid_indices):
        pixel_features = F.normalize(rendered_features[:, masks[mask_index]].transpose(0, 1), dim=1)
        positive = detached_prototypes[index_position]
        negatives = [
            prototype
            for other_position, prototype in enumerate(detached_prototypes)
            if other_position != index_position
        ]
        if detached_background is not None:
            negatives.append(detached_background)

        if not negatives:
            mask_losses.append(_zero_loss(pixel_features))
            continue

        positive_logits = torch.sum(pixel_features * positive.unsqueeze(0), dim=1, keepdim=True) / temperature
        negative_logits = pixel_features @ torch.stack(negatives, dim=0).transpose(0, 1) / temperature
        logits = torch.cat((positive_logits, negative_logits), dim=1)
        targets = torch.zeros(pixel_features.shape[0], dtype=torch.long, device=pixel_features.device)
        mask_losses.append(F.cross_entropy(logits, targets))

    return InstanceLossResult(
        loss=torch.stack(mask_losses).mean(),
        valid_masks=len(valid_indices),
        used_background=background_prototype is not None,
    )


def semantic_mask_alignment_loss(
    masks: torch.Tensor,
    labels: torch.Tensor,
    label_features: torch.Tensor,
    rendered_features: torch.Tensor,
) -> SemanticLossResult:
    if masks.ndim != 3:
        raise ValueError("`masks` must have shape [N, H, W]")
    if labels.ndim != 1:
        raise ValueError("`labels` must have shape [N]")
    if rendered_features.ndim != 3:
        raise ValueError("`rendered_features` must have shape [C, H, W]")
    if masks.shape[0] != labels.shape[0]:
        raise ValueError("Mask count and label count must match")
    if label_features.ndim != 2:
        raise ValueError("`label_features` must have shape [L, C]")
    if label_features.shape[1] != rendered_features.shape[0]:
        raise ValueError("Label feature dimension must match rendered semantic feature dimension")
    if labels.numel() > 0 and (labels.min().item() < 0 or labels.max().item() >= label_features.shape[0]):
        raise ValueError("Label index out of range")

    mask_losses: list[torch.Tensor] = []
    for mask_index, mask in enumerate(masks):
        if not bool(mask.any()):
            continue
        mask_prototype = _normalized_mask_mean(rendered_features, mask)
        target_prototype = F.normalize(label_features[labels[mask_index]], dim=0)
        cosine = F.cosine_similarity(mask_prototype.unsqueeze(0), target_prototype.unsqueeze(0), dim=1)
        mask_losses.append(1.0 - cosine.squeeze(0))

    if not mask_losses:
        return SemanticLossResult(loss=_zero_loss(rendered_features), valid_masks=0)

    return SemanticLossResult(loss=torch.stack(mask_losses).mean(), valid_masks=len(mask_losses))
