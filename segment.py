import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from PIL import Image
import cv2
import torch
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
    return detections

def class_to_feature(classes, dim=32, device='cpu'):
    ...
def words_to_tensors(word_list, dim=32, device='cpu'):
    """使用正弦余弦函数生成确定性向量"""
    vectors = torch.zeros((len(word_list), dim), device=device)
    
    for i, word in enumerate(word_list):
        hash_val = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        
        frequencies = np.random.randn(dim // 2) * 10
        phases = np.random.rand(dim // 2) * 2 * np.pi
        
        for j in range(dim // 2):
            vectors[i, 2*j] = torch.tensor(np.sin(frequencies[j] + phases[j]))
            vectors[i, 2*j + 1] = torch.tensor(np.cos(frequencies[j] + phases[j]))
    
    # 添加归一化步骤
    norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
    normalized_vectors = vectors / norms
    
    return normalized_vectors

@hydra.main(config_path="configs", config_name="segment", version_base=None)
def main(cfg: DictConfig):
    args = cfg.segment
    prepare_output_folder(args)
    sam, dino = load_models(args)

    torch.save(words_to_tensors(args.classes), os.path.join(args.labels_path, f'label_features.pt'))

    images_name = sorted([e for e in os.listdir(args.images_path) if e.endswith('.jpg')])
    progress_bar = tqdm(images_name)
    for image_name in progress_bar:
        if args.progress_path:
            with open(args.progress_path, 'w') as f:
                f.write(str((progress_bar.n+1)*100//progress_bar.total))

        image = cv2.cvtColor(cv2.imread(os.path.join(args.images_path, image_name)), cv2.COLOR_BGR2RGB)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if args.downsample != 1 and args.downsample_type == 'image':
            rotated_image = cv2.resize(rotated_image,dsize=(rotated_image.shape[1] // args.downsample, rotated_image.shape[0] // args.downsample),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)

        detections = object_detection(args, dino, rotated_image)
        # if detections.xyxy.shape[0] == 0:
        #     continue

        # convert detections to masks
        detections.mask = segment(
            args,
            sam_predictor=sam,
            image=rotated_image,
            xyxy=detections.xyxy
        )

        if args.downsample != 1 and args.downsample_type == 'mask':
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

if __name__ == "__main__":
    main()