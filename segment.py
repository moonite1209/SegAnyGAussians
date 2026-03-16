import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
from typing import List, Optional, Protocol, Dict, Any
from dataclasses import dataclass

import cv2
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from PIL import Image
import logging

from transformers import SamModel, SamProcessor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv
import hydra
from omegaconf import DictConfig, OmegaConf
from saga_config import DatasetConfig, SegmentAppConfig, SegmentConfig
from hydra.utils import instantiate

log = logging.getLogger(__name__)

# ============ 常量定义 ============

DEFAULT_FEATURE_DIM = 16  # 类别特征维度（用于后续训练的语义特征）
DEFAULT_ROTATION = 90     # 默认旋转角度（顺时针）
HASH_ALGORITHM = 'md5'    # 哈希算法（用于确定性特征生成）

SAM_ARCHITECTURES = ['vit_h', 'vit_l', 'vit_b']

# ============ 数据结构 ============

@dataclass
class DetectionResult:
    """
    统一的检测结果。

    Attributes:
        boxes: (N, 4) xyxy格式的边界框数组
        confidences: (N,) 置信度分数数组
        class_ids: (N,) 类别ID数组
    """
    bboxes: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    classes: List[str]

    def is_empty(self) -> bool:
        """检查是否检测到任何对象。"""
        return len(self.class_ids) == 0


@dataclass
class SegmentationResult:
    """
    统一的分割结果。

    Attributes:
        masks: (N, H, W) 布尔掩码数组
        class_ids: (N,) 类别ID数组
        boxes: (N, 4) xyxy格式的边界框数组（可选）
        confidences: (N,) 置信度分数数组（可选）
    """
    masks: np.ndarray
    class_ids: np.ndarray
    classes: List[str]
    bboxes: Optional[np.ndarray] = None
    confidences: Optional[np.ndarray] = None

    def is_empty(self) -> bool:
        """检查是否检测到任何对象。"""
        return len(self.class_ids) == 0
    
    def rotate(self, rotateCode: int) -> None:
        """
        旋转结果中的掩码和边界框。

        Args:
            rotateCode: 旋转代码
        """
        if rotateCode == cv2.ROTATE_90_COUNTERCLOCKWISE:
            # 获取原始尺寸 (N, H, W)
            N, H, W = self.masks.shape

            # 1. 旋转掩码 (逆时针 90度)
            # np.rot90 k=1 是逆时针
            self.masks = np.rot90(self.masks, k=1, axes=(1, 2)).copy()

            # 2. 旋转边界框
            # 逆时针 90度变换公式: (x, y) -> (y, W - x)
            # 新图尺寸变为 (W, H)
            if self.bboxes is not None:
                new_bboxes = self.bboxes.copy()
                # 原 bbox: [x1, y1, x2, y2]
                # x1, y1 是左上角; x2, y2 是右下角
                # 新 x1 = 原 y1
                new_bboxes[:, 0] = self.bboxes[:, 1]
                # 新 y1 = W - 原 x2 (注意：这里要减去最大的 x，即 x2，才能得到最小的 y)
                new_bboxes[:, 1] = W - self.bboxes[:, 2]
                # 新 x2 = 原 y2
                new_bboxes[:, 2] = self.bboxes[:, 3]
                # 新 y2 = W - 原 x1
                new_bboxes[:, 3] = W - self.bboxes[:, 0]
                self.bboxes = new_bboxes


# ============ 模型接口 ============

class Detector(Protocol):
    """检测器接口 - 所有检测模型必须实现。"""

    def detect(
        self,
        image: np.ndarray,
        classes: List[str],
        **kwargs
    ) -> DetectionResult:
        """
        检测图像中的对象。

        Args:
            image: RGB图像 (H, W, C)
            classes: 类别名列表
            **kwargs: 模型特定参数

        Returns:
            DetectionResult
        """
        ...


class BBoxSegmenter(Protocol):
    """分割器接口 - 所有分割模型必须实现。"""

    def segment(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        为边界框生成掩码。

        Args:
            image: RGB图像 (H, W, C)
            bboxes: 边界框 (N, 4) xyxy格式
            **kwargs: 模型特定参数

        Returns:
            masks: (N, H, W) bool数组
        """
        ...


class OVSegmenter(Protocol):
    """开集分割器接口 - 一体化模型。"""

    def segment(
        self,
        image: np.ndarray,
        classes: List[str],
        **kwargs
    ) -> SegmentationResult:
        """
        直接从图像和类别生成分割结果。

        Args:
            image: RGB图像 (H, W, C)
            classes: 类别名列表
            **kwargs: 模型特定参数

        Returns:
            SegmentationResult
        """
        ...


# ============ 模型实现 ============

class GroundingDINODetector(Detector):
    """GroundingDINO检测器实现。"""

    # 分隔符 token IDs (BERT tokenizer)
    SEP_TOKEN_IDS = {101, 102, 1012}  # [CLS], [SEP], .

    def __init__(
        self,
        model_id = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.35
    ):
        """
        初始化GroundingDINO检测器。

        Args:
            model_id: HuggingFace 模型ID
            device: 运行设备
            box_threshold: 边界框置信度阈值
            text_threshold: 文本置信度阈值
        """
        self.model_id = model_id
        self.device = device

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        log.info(f"GroundingDINO loaded from {model_id}")

    def _remove_cross_segment_activation(
            self,
            outputs,
            input_ids,
        ):
        """
        修改 outputs.logits，移除跨段激活，解决交叉查询问题。
        对于每个 query，只保留最高激活的段内的 logits，将其他段的 logits 置为负无穷。
        这样可以确保每个检测结果只对应一个独立的短语段。
        参考: https://github.com/IDEA-Research/GroundingDINO/issues/85
        """
        import copy

        # 深拷贝 outputs
        outputs = copy.deepcopy(outputs)

        batch_logits = outputs.logits  # (batch_size, num_queries, seq_len_padded) e.g., (B, 900, 256)
        device = batch_logits.device
        
        # 获取 logits 的序列长度 (通常是 256)
        padded_seq_len = batch_logits.shape[-1]
        
        # 预先将 SEP ID 转为 Tensor
        sep_token_ids_tensor = torch.tensor(list(self.SEP_TOKEN_IDS), device=device)

        for batch_idx in range(batch_logits.shape[0]):
            token_ids = input_ids[batch_idx]  # (actual_seq_len,) e.g., 46
            
            # 1. 找到分隔符位置 (基于实际输入 input_ids)
            sep_mask = torch.isin(token_ids, sep_token_ids_tensor)
            sep_positions = torch.where(sep_mask)[0]

            if len(sep_positions) < 2:
                continue

            num_segments = len(sep_positions) - 1

            # 2. 创建段掩码 (基于 Logits 的 padded 长度)
            # 关键修复：使用 padded_seq_len (256) 而不是 token_ids.shape[0] (46)
            segment_ids = torch.full((padded_seq_len,), -1, dtype=torch.long, device=device)
            
            for i in range(num_segments):
                # 跳过分隔符本身，只标记段内的 token 内容
                start, end = sep_positions[i] + 1, sep_positions[i + 1]
                if start < end:
                    segment_ids[start:end] = i

            # 3. 向量化计算 Aggregation
            query_logits = batch_logits[batch_idx]  # (num_queries, 256)
            
            # 扩展 segment_ids: (num_queries, 256)
            segment_ids_expanded = segment_ids.unsqueeze(0).expand(query_logits.shape[0], -1)

            # valid_mask: 只有属于有效段落的部分参与计算 (-1 表示 Padding, CLS, SEP)
            valid_mask = segment_ids_expanded != -1
            
            # 准备 Scatter 数据
            segment_ids_for_scatter = segment_ids_expanded+1

            # Mask 掉非段落区域的值 (置为 0)
            masked_logits = query_logits.masked_fill(~valid_mask, 0.0)
            ones = valid_mask.float()

            # 计算 Sum 和 Count
            segment_sum = torch.zeros(query_logits.shape[0], num_segments+1, device=device)
            segment_count = torch.zeros(query_logits.shape[0], num_segments+1, device=device)
            
            segment_sum.scatter_add_(1, segment_ids_for_scatter, masked_logits)
            segment_count.scatter_add_(1, segment_ids_for_scatter, ones)

            # 计算 Mean (防止除零)
            valid_segment_mask = segment_count > 0
            segment_mean = torch.full_like(segment_sum, float('-inf')) 
            segment_mean[valid_segment_mask] = segment_sum[valid_segment_mask] / segment_count[valid_segment_mask]
            segment_mean = segment_mean[:, 1:]
            # 4. 找到每个 query 的最佳段
            best_segment_idx = torch.argmax(segment_mean, dim=1)  # (num_queries,)

            # 5. 应用掩码
            # 保留：(是最佳段落)
            # 剔除：(是段落内容) AND (不是最佳段落) OR (是特殊符号/Padding)
            is_seq = segment_ids_expanded == -1
            is_segment = segment_ids_expanded != -1
            is_best_segment = segment_ids_expanded == best_segment_idx.unsqueeze(1)
            
            batch_logits[batch_idx][~is_best_segment] = float('-inf')

        outputs.logits = batch_logits
        return outputs

    def detect(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """
        执行对象检测。

        Args:
            image: RGB图像 (H, W, C)
            classes: 类别名列表
            box_threshold: 边界框置信度阈值（可选，覆盖初始化值）
            text_threshold: 文本置信度阈值（可选，覆盖初始化值）

        Returns:
            DetectionResult - 包含所有检测到的对象，每个对象对应一个类别ID
        """
        # 使用传入的阈值或初始化时的阈值
        box_thresh = box_threshold if box_threshold is not None else self.box_threshold
        text_thresh = text_threshold if text_threshold is not None else self.text_threshold

        # 准备输入 - 使用句号分隔的字符串格式
        # text_prompt = ". ".join(classes) + "."
        text_prompt = classes
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 移除跨段激活，解决交叉查询问题
        outputs = self._remove_cross_segment_activation(outputs, inputs.input_ids)

        # 使用库的后处理方法
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[image.shape[:2]]
        )

        # post_process_grounded_object_detection 返回列表，每个图像对应一个结果
        if not results or len(results) == 0:
            return DetectionResult(
                bboxes=np.zeros((0, 4), dtype=np.float32),
                confidences=np.zeros(0, dtype=np.float32),
                class_ids=np.zeros(0, dtype=np.int64)
            )

        result = results[0]  # 单图像情况

        # 提取边界框和置信度
        bboxes = result.get("boxes", torch.empty(0, 4)).cpu().numpy()
        confidences = result.get("scores", torch.empty(0)).cpu().numpy()
        text_labels = result.get("text_labels", [])
        labels = result.get("labels", [])

        # 如果没有检测结果，返回空结果
        if len(bboxes) == 0:
            return DetectionResult(
                bboxes=np.zeros((0, 4), dtype=np.float32),
                confidences=np.zeros(0, dtype=np.float32),
                class_ids=np.zeros(0, dtype=np.int64),
                classes=[]
            )

        # 将文本标签映射回类别索引
        # 由于 GroundingDINO 只检测输入的类别，text_labels 应该都在 classes 中
        class_ids = np.array([classes.index(text_label) for text_label in text_labels], dtype=np.int64)

        return DetectionResult(
            bboxes=bboxes.astype(np.float32),
            confidences=confidences.astype(np.float32),
            class_ids=class_ids,
            classes=text_labels
        )


class SAMSegmenter(BBoxSegmenter):
    """SAM分割器实现。"""

    def __init__(
        self,
        model_id: str = "facebook/sam-vit-base",
        device: str = 'cuda'
    ):
        """
        初始化SAM分割器。

        Args:
            model_id: HuggingFace 模型ID
            device: 运行设备
        """
        self.device = device
        self.model_id = model_id
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_id)
        log.info(f"SAM loaded from {model_id}")

    def segment(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        为边界框生成掩码。

        Args:
            image: RGB图像 (H, W, C)
            bboxes: 边界框 (N, 4) xyxy格式
            **kwargs: 未使用（为接口兼容性保留）

        Returns:
            masks: (N, H, W) bool数组
        """
        B, N, H, W = 1, bboxes.shape[0], image.shape[0], image.shape[1]
        if len(bboxes) == 0:
            return np.zeros((0, H, W), dtype=bool)

        # SAM processor 支持直接输入 numpy 数组
        # input_boxes 需要是嵌套列表格式: [[x1, y1, x2, y2], ...]
        inputs = self.processor(
            [image],
            input_boxes=[bboxes.tolist()],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.pred_masks: [B, N, 3, MH, MW] - 3个候选掩码
        # outputs.iou_scores: [B, N, 3] - 每个掩码的IoU分数
        # 根据 IoU 分数选择最佳掩码
        _, _, _, MH, MW = outputs.pred_masks.shape
        best_indices = torch.argmax(outputs.iou_scores, dim=2, keepdim=True)  # [B, N, 1]

        best_masks = torch.gather(outputs.pred_masks, 2, best_indices.view(B, N, 1, 1, 1).expand(B, N, 1, MH, MW))

        # post_process_masks 将掩码调整到原始图像尺寸
        masks_list = self.processor.image_processor.post_process_masks(
            best_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        if not masks_list or len(masks_list) == 0:
            return np.zeros((0, H, W), dtype=bool)

        masks = masks_list[0].squeeze(1)  # (N, H, W)

        # 转换为布尔类型
        masks = (masks > 0.5).numpy().astype(bool)

        return masks

# ============ 组合模型 ============

class OVDetectSegmenter(OVSegmenter):
    """组合检测器和分割器，支持通过 hydra 配置灵活组合。

    hydra instantiate 会递归实例化所有嵌套配置，所以传入的
    detector 和 segmenter 已经是实例化好的对象。
    """

    def __init__(
        self,
        detector: Detector,
        segmenter: BBoxSegmenter,
    ):
        """
        初始化组合分割器。

        Args:
            detector: 已实例化的检测器对象（hydra 自动递归实例化）
            segmenter: 已实例化的分割器对象（hydra 自动递归实例化）
        """
        self.detector = detector
        self.segmenter = segmenter
        log.info("OVDetectSegmenter initialized")

    def segment(
        self,
        image: np.ndarray,
        classes: List[str],
        **kwargs
    ) -> SegmentationResult:
        """
        使用 GroundingDINO 检测对象，然后使用 SAM 生成分割掩码。

        Args:
            image: RGB图像 (H, W, C)
            classes: 类别名列表
            **kwargs: 额外参数（box_threshold, text_threshold）

        Returns:
            SegmentationResult
        """
        # 步骤1: 使用 GroundingDINO 检测对象
        detection_result = self.detector.detect(
            image,
            classes,
            box_threshold=kwargs.get('box_threshold'),
            text_threshold=kwargs.get('text_threshold')
        )

        # 如果没有检测到任何对象，返回空结果
        if detection_result.is_empty():
            H, W = image.shape[:2]
            return SegmentationResult(
                masks=np.zeros((0, H, W), dtype=bool),
                class_ids=np.zeros(0, dtype=np.int64),
                classes=[],
                bboxes=np.zeros((0, 4), dtype=np.float32),
                confidences=np.zeros(0, dtype=np.float32)
            )

        # 步骤2: 使用 SAM 为每个边界框生成分割掩码
        masks = self.segmenter.segment(image, detection_result.bboxes)

        # 返回完整的分割结果
        return SegmentationResult(
            masks=masks,
            class_ids=detection_result.class_ids,
            classes=detection_result.classes,
            bboxes=detection_result.bboxes,
            confidences=detection_result.confidences
        )


# ============ 辅助函数 ============

def words_to_tensors(word_list: List[str], dim: int = 32, device: str = 'cpu') -> torch.Tensor:
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

def prepare_output_folder(masks_dir: str, labels_dir: str, rgb_masks_dir: Optional[str] = None):
    """
    准备输出文件夹。

    Args:
        masks_dir: 掩码保存目录
        labels_dir: 标签保存目录
        rgb_masks_dir: RGB掩码保存目录（可选）
    """
    Path(masks_dir).mkdir(parents=True, exist_ok=True)
    Path(labels_dir).mkdir(parents=True, exist_ok=True)
    if rgb_masks_dir:
        Path(rgb_masks_dir).mkdir(parents=True, exist_ok=True)

def segment_one_image(
    args: SegmentConfig,
    image_path: Path,
    ovsegmenter: OVSegmenter
) -> SegmentationResult:
    """
    处理单张图像。

    Args:
        args: 配置参数
        image_path: 图像文件路径
        ovsegmenter: 加载的OVSegmenter

    Returns:
        SegmentationResult
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # rotate image, workaround with farsee dataset image orientation
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # 分割
    result = ovsegmenter.segment(image, list(args.classes))

    result.rotate(cv2.ROTATE_90_COUNTERCLOCKWISE)

    return result


def save_result(
    result: SegmentationResult,
    image_path: Path,
    masks_dir: Path,
    labels_dir: Path,
    rgb_masks_dir: Optional[Path] = None
):
    """
    保存分割结果。

    Args:
        result: 分割结果（包含 classes 字段）
        image_path: 原始图像路径
        masks_dir: 掩码保存目录
        labels_dir: 标签保存目录
        rgb_masks_dir: RGB掩码保存目录（可选）
    """
    base_name = image_path.stem

    # 保存掩码
    torch.save(torch.from_numpy(result.masks), masks_dir / f'{base_name}.pt')

    # 保存标签
    torch.save(
        torch.tensor(result.class_ids, dtype=torch.int64),
        labels_dir / f'{base_name}.pt'
    )

    # 保存RGB掩码（可选）
    if not rgb_masks_dir:
        return
    # 读取原始图像用于可视化
    original_image = cv2.imread(str(image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    if not result.is_empty():
        # 创建检测结果对象用于可视化
        detections = sv.Detections(
            xyxy=result.bboxes,
            mask=result.masks,
            class_id=result.class_ids,
            confidence=result.confidences if result.confidences is not None else np.ones(len(result.class_ids))
        )

        # 创建RGB掩码
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()

        # 生成标签：直接使用 result.classes（检测到的每个对象对应的类别名）
        labels = None
        if result.classes is not None and result.confidences is not None:
            labels = [
                f"{clz} {conf:.2f}"
                for clz, conf in zip(result.classes, result.confidences)
            ]

        annotated_image = mask_annotator.annotate(scene=original_image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        if labels:
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    else:
        # 无检测结果，直接保存原始图像
        annotated_image = original_image

    cv2.imwrite(
        str(rgb_masks_dir / f'{base_name}.jpg'),
        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    )

# ============ 主函数 ============

@hydra.main(config_path="configs", config_name="segment", version_base=None)
def main(cfg: DictConfig):
    """
    主入口点：执行图像分割流水线。

    流程：
    1. 准备输出文件夹
    2. 创建流水线
    3. 生成类别特征
    4. 处理所有图像
    5. 保存结果
    """
    app_cfg = SegmentAppConfig(**OmegaConf.to_container(cfg, resolve=True))
    dataset: DatasetConfig = app_cfg.dataset
    args: SegmentConfig = app_cfg.segment
    log.info("Starting segmentation pipeline")

    # 准备输出文件夹
    prepare_output_folder(
        args.masks_dir,
        args.labels_dir,
        args.rgb_masks_dir
    )

    # 创建分割器（使用 hydra instantiate）
    log.info("Creating segmenter...")
    ovsegmenter = instantiate(args.ovsegmenter.dump_model(by_alias=True, exclude_none=True))

    # 生成类别特征
    log.info("Generating class features...")
    label_features = words_to_tensors(args.classes, dim=DEFAULT_FEATURE_DIM)
    torch.save(
        label_features,
        Path(args.label_features_path)
    )

    # 处理图像
    images_dir = Path(dataset.images_path)
    image_files = sorted(images_dir.glob('*.jpg'))

    if len(image_files) == 0:
        log.warning(f"No .jpg images found in {images_dir}")
        return

    log.info(f"Found {len(image_files)} images to process")

    success_count = 0
    for image_path in tqdm(image_files, desc="Segmenting"):
        try:
            result = segment_one_image(args, image_path, ovsegmenter)
            save_result(
                result,
                image_path,
                masks_dir=Path(args.masks_dir),
                labels_dir=Path(args.labels_dir),
                rgb_masks_dir=Path(args.rgb_masks_dir) if args.rgb_masks_dir else None
            )
            success_count += 1
        except Exception as e:
            log.error(f"Error processing {image_path.name}: {str(e)}")
            continue

    log.info(f"Successfully processed {success_count}/{len(image_files)} images")



if __name__ == "__main__":
    main()
