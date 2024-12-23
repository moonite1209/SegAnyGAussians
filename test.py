import torch
from torchvision.utils import save_image
import numpy as np
import os
import shutil
import torch.nn.functional as F

def sam_masks_rgb():
    spath = 'data/temp/shenyang/sam_masks'
    dpath = 'data/temp/shenyang/sam_masks_rgb'
    os.makedirs(dpath, exist_ok=True)
    for masks_name in os.listdir(spath):
        print(os.path.join(spath, masks_name))
        masks=torch.load(os.path.join(spath, masks_name))
        n, h, w = masks.shape
        colormap = torch.rand((n+1, 3)).cuda()
        colormap[-1]=0
        masks = sorted(masks, key=lambda m: m.sum())
        segmentmap=torch.full(masks[0].shape, -1).cuda()
        for idx, mask in enumerate(masks):
            segmentmap[mask]=idx
        save_image(colormap[segmentmap].permute(2,0,1).cpu(), os.path.join(dpath, masks_name.split('.')[0]+'.jpg'))

def pick_image():
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
    # sparse_path = '/mnt/d/BaiduNetdiskDownload/nanfeng/分块sfm数据/chunks/layer_3/chunk_003_001/sparse/0/'
    sparse_path = 'data/temp/nanfeng/sparse/0/'
    image_path = '/mnt/d/BaiduNetdiskDownload/nanfeng/图像集/images/'
    dst_path = 'data/temp/nanfeng/images'
    images = read_extrinsics_binary(os.path.join(sparse_path, 'images.bin'))
    cameras = read_intrinsics_binary(os.path.join(sparse_path, 'cameras.bin'))
    for image in images.values():
        shutil.copy(os.path.join(image_path, image.name), dst_path)

def test_clip():
    from transformers import CLIPModel, CLIPProcessor
    from segment_anything import (SamAutomaticMaskGenerator, SamPredictor, sam_model_registry)
    from PIL import Image
    import cv2
    device = 'cuda'
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    sam = sam_model_registry['vit_h'](checkpoint='./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth').to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        box_nms_thresh=0.7,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    image = Image.open('data/temp/nanfeng/images/00000000001-00000001113-A01113.jpg')
    records = mask_generator.generate(np.array(image))
    def get_bbox(mask: np.ndarray):
        # 查找掩码中的 True 元素的索引
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        # 如果没有 True 元素，则返回全零的边界框
        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)
        
        # 获取边界框的上下左右边界
        x_min, x_max = np.where(rows)[0][[0, -1]] # h
        y_min, y_max = np.where(cols)[0][[0, -1]] # w
        
        # 返回边界框
        return (x_min, y_min, x_max + 1 - x_min, y_max + 1 - y_min) # x, y, h, w
    def get_entity_image(image: np.ndarray, mask: np.ndarray)->np.ndarray:
        if mask.sum()==0:
            return np.zeros((224,224,3), dtype=np.uint8)
        image = image.copy()
        # crop by bbox
        x,y,h,w = get_bbox(mask)
        image[~mask] = np.zeros(3, dtype=np.uint8) #分割区域外为白色
        image = image[x:x+h, y:y+w, ...] #将img按分割区域bbox裁剪
        # pad to square
        l = max(h,w)
        paded_img = np.zeros((l, l, 3), dtype=np.uint8)
        if h > w:
            paded_img[:,(h-w)//2:(h-w)//2 + w, :] = image
        else:
            paded_img[(w-h)//2:(w-h)//2 + h, :, :] = image
        paded_img = cv2.resize(paded_img, (224,224))
        return paded_img
    
    entity = [get_entity_image(np.array(image), record['segmentation']) for record in records]
    inputs = clip_processor(images=entity, return_tensors='pt')
    inputs = inputs.to(clip_model.device)
    semantics = clip_model.get_image_features(**inputs)
    semantics = F.normalize(semantics,dim=-1).detach().cpu().numpy()
    np.save('entity.npy', entity)
    np.save('semantics.npy', semantics)


def main():
    # sam_masks_rgb()
    # pick_image()
    test_clip()

if __name__ =='__main__':
    main()