import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import hashlib
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from groundingdino.util.inference import Model
import supervision as sv
import hydra
from omegaconf import DictConfig, OmegaConf

def load_models(args):
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    sam_predictor = SamPredictor(sam)
    grounding_dino_model = Model(model_config_path=args.groundingdino_config_path, model_checkpoint_path=args.groundingdino_checkpoint_path)
    return sam_predictor, grounding_dino_model

def prepare_output_folder(args):
    masks_path = args.masks_path
    os.makedirs(masks_path, exist_ok=True)
    labels_path = args.labels_path
    os.makedirs(labels_path, exist_ok=True)
    if args.progress_path:
        os.makedirs(os.path.dirname(args.progress_path), exist_ok=True)
    if args.rgb_masks_path:
        os.makedirs(args.rgb_masks_path, exist_ok=True)

def write_progress(progress_path, value):
    if progress_path:
        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        with open(progress_path, 'w') as f:
            f.write(str(value))
def segment(args, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    H,W,C = image.shape
    sam_predictor.set_image(image)
    result_masks = np.zeros((0,H,W), dtype='b1')
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks = np.concat([result_masks, masks[index][None]], axis=0)
    return result_masks
def rotate_detections_90_ccw(detections, image_width, image_height):
    # 提取原始检测框坐标
    x1, y1, x2, y2 = detections.xyxy.T

    # 计算旋转后的坐标
    new_x1 = y1
    new_y1 = image_width - x2
    new_x2 = y2
    new_y2 = image_width - x1

    # 更新检测框坐标
    rotated_boxes = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
    detections.xyxy = rotated_boxes

    return detections

def object_detection(args, dino, image):
    detections = dino.predict_with_classes(
        image=image,
        classes=args.classes,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        args.nms_threshold
    ).numpy().tolist()
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    
    # 过滤掉class_id为None的检测结果
    valid_idx = [i for i, c in enumerate(detections.class_id) if c is not None]
    detections.xyxy = detections.xyxy[valid_idx]
    detections.confidence = detections.confidence[valid_idx]
    detections.class_id = detections.class_id[valid_idx]
    
    # 将class_id转换为np.int64类型
    if len(detections.class_id) > 0:
        detections.class_id = np.array(detections.class_id, dtype=np.int64)
    
    return detections

def class_to_feature(classes, dim=32, device='cpu'):
    ...
def words_to_tensors(word_list, dim=32, device='cpu'):
    """
    Generate semantic features with equiangular distribution on unit hypersphere.
    Uses regular simplex projection to maximize minimum inter-point distance.

    This method produces features where all pairwise distances are equal and
    maximally separated, which is optimal for semantic class discrimination.

    For n classes in dim dimensions, requires n <= dim + 1.

    Args:
        word_list: List of class names
        dim: Feature dimension (default: 32)
        device: torch device

    Returns:
        Tensor of shape (len(word_list), dim) with normalized features
    """
    num_classes = len(word_list)
    
    # --- 策略 1: 解析解 (SVD分解) ---
    # 适用于维度足够容纳单纯形的情况 (dim >= N - 1)
    # 这种方法生成的任意两点间余弦相似度恒定为 -1/(N-1)
    if dim >= num_classes - 1:
        # 1. 构建中心化矩阵 M = I - 1/N * J
        # M 的每一行代表一个顶点，但它们位于 N 维空间
        # 通过 SVD 降维到 N-1 维
        M = torch.eye(num_classes, device=device) - (1.0 / num_classes)
        
        # 2. SVD 分解
        # M 是半正定矩阵，秩为 N-1
        U, S, _ = torch.linalg.svd(M)
        
        # 3. 提取前 N-1 个特征向量并缩放
        # 我们取 U 的部分列作为特征，此时行向量模长为 sqrt((N-1)/N)
        # 为了归一化，我们需要除以这个模长，或者直接最后做一次 F.normalize
        # 理论上只取前 num_classes - 1 列即可构建单纯形
        feats = U[:, :num_classes - 1]
        
        # 4. 填充零以匹配目标维度 dim
        # 当前 feats 形状为 (N, N-1)，需要 pad 到 (N, dim)
        pad_size = dim - (num_classes - 1)
        if pad_size > 0:
            feats = F.pad(feats, (0, pad_size), "constant", 0)
            
    # --- 策略 2: 优化解 (梯度下降) ---
    # 适用于维度不足的情况 (dim < N - 1)，即强行把 N 个点塞进低维空间
    else:
        # 使用确定性种子确保可重现性
        words = sorted(word_list)
        seed = int(hashlib.md5("|".join(words).encode()).hexdigest()[:8], 16)
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        # 初始化随机向量（使用确定性生成器）
        feats = torch.randn(num_classes, dim, device=device, generator=g)
        feats.requires_grad = True

        # 使用优化器调整位置
        optimizer = torch.optim.Adam([feats], lr=0.1)
        
        # 迭代寻找最小化余弦相似度（即最大化角度）的布局
        # 这种布局称为 ETF (Equiangular Tight Frame)
        for _ in range(200):
            optimizer.zero_grad()
            
            # 归一化
            feats_norm = F.normalize(feats, p=2, dim=1)
            
            # 计算 Gram 矩阵 (余弦相似度矩阵)
            gram = torch.mm(feats_norm, feats_norm.t())
            
            # 目标：让非对角线元素的平方和最小（即让所有向量尽可能正交或反向）
            # 或者逼近 Welch Bound (理论下界)
            # 简单的损失函数：最小化 Gram 矩阵与单位阵的差异 (除了对角线)
            target = torch.eye(num_classes, device=device)
            
            # 仅优化非对角部分，使其尽可能小（趋向于 -1/(N-1) 或 Welch Bound）
            # 这里使用 Frobenius 范数作为 loss 推动特征分离
            loss = (gram - target).pow(2).mean()
            
            loss.backward()
            optimizer.step()
        
        # 最后关闭梯度
        feats = feats.detach()

    # 最后确保严格归一化
    feats = F.normalize(feats, p=2, dim=1)
    
    return feats

@hydra.main(config_path="configs", config_name="segment", version_base=None)
def main(cfg: DictConfig):
    args = cfg.segment
    prepare_output_folder(args)
    torch.save(words_to_tensors(args.classes, 16), os.path.join(args.labels_path, f'label_features.pt'))
    sam, dino = load_models(args)


    images_name = sorted([e for e in os.listdir(args.images_path) if e.endswith('.jpg')])
    progress_bar = tqdm(images_name)
    for image_name in progress_bar:
        write_progress(args.progress_path, (progress_bar.n+1)*100//progress_bar.total)

        image = cv2.cvtColor(cv2.imread(os.path.join(args.images_path, image_name)), cv2.COLOR_BGR2RGB)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if args.downsample != 1 and args.downsample_type == 'image':
            rotated_image = cv2.resize(rotated_image,dsize=(rotated_image.shape[1] // args.downsample, rotated_image.shape[0] // args.downsample),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        with torch.no_grad():
            detections = object_detection(args, dino, rotated_image)

        # convert detections to masks
        if detections.xyxy.shape[0] == 0:
            # 如果没有检测到物体，生成空的mask数组
            H, W = rotated_image.shape[:2]
            if args.downsample != 1 and args.downsample_type == 'mask':
                H, W = H // args.downsample, W // args.downsample
            detections.mask = np.zeros((0, H, W), dtype='b1')
            detections.class_id = np.array([], dtype=np.int64)
        else:
            # 正常检测物体并生成mask
            with torch.no_grad():
                detections.mask = segment(
                    args,
                    sam_predictor=sam,
                    image=rotated_image,
                    xyxy=detections.xyxy
                )

        if args.downsample != 1 and args.downsample_type == 'mask' and detections.mask.shape[0] > 0:
            H,W = rotated_image.shape[0] // args.downsample, rotated_image.shape[1] // args.downsample
            mask_list = np.zeros((0,H,W), dtype='b1')
            for i, mask in enumerate(detections.mask):
                mask_score = torch.from_numpy(mask).float()
                mask_score = torch.nn.functional.interpolate(mask_score[None, None, ...], size=(rotated_image.shape[0] // args.downsample, rotated_image.shape[1] // args.downsample) , mode='bilinear', align_corners=False).squeeze()
                mask_score[mask_score >= 0.5] = 1
                mask_score[mask_score != 1] = 0
                mask_score = mask_score.bool().numpy()
                mask_list = np.concat([mask_list, mask_score[None]], axis=0)
            detections.mask = mask_list

        torch.save(torch.from_numpy(detections.mask).permute(0, 2, 1).flip(1), os.path.join(args.masks_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.pt')) # bool[masks, h, w]
        torch.save(torch.tensor(detections.class_id, dtype=torch.int64), os.path.join(args.labels_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.pt')) # int[masks]

        if args.rgb_masks_path:
            if detections.xyxy.shape[0] > 0:  # 只有当有检测结果时才生成带注释的图像
                box_annotator = sv.BoxAnnotator()
                mask_annotator = sv.MaskAnnotator()
                label_annotator = sv.LabelAnnotator()
                labels = [
                    f"{args.classes[class_id.item()]} {confidence.item():0.2f}" 
                    for _, _, confidence, class_id, _, _ 
                    in detections]
                annotated_image = mask_annotator.annotate(scene=rotated_image.copy(), detections=detections)
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
                cv2.imwrite(os.path.join(args.rgb_masks_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.jpg'), cv2.cvtColor(cv2.rotate(annotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_RGB2BGR))
            else:
                # 如果没有检测结果，直接保存旋转后的原始图像
                cv2.imwrite(os.path.join(args.rgb_masks_path, f'{os.path.splitext(os.path.basename(image_name))[0]}.jpg'), cv2.cvtColor(cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()