import torch
from torchvision.utils import save_image
import os

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
        save_image(colormap[segmentmap].permute(2,0,1).cpu(), os.path.join(dpath, os.path.splitext(masks_name)[0], 'jpg'))



def main():
    sam_masks_rgb()

if __name__ =='__main__':
    main()