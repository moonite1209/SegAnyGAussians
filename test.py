import torch
from torchvision.utils import save_image
import os
import shutil

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

def main():
    # sam_masks_rgb()
    pick_image()

if __name__ =='__main__':
    main()