import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
from torchvision.utils import save_image
import numpy as np
import shutil
import torch.nn.functional as F
from PIL import Image

def pth_to_json():
    import json
    label_to_class = {0:'plant', 1:'chair', 2:'table', 3:'object'}
    point_ins_labels_path = 'temp/tys/point_ins_labels.pth'
    point_labels_path = 'temp/tys/point_labels.pth'
    point_ins_labels = torch.load(point_ins_labels_path, weights_only=True)
    point_labels = torch.load(point_labels_path, weights_only=True)
    output = dict()
    output['point_instance_labels'] = point_ins_labels.tolist()
    output['instances'] = [{'class': label_to_class[l]} for l in point_labels.tolist()]
    with open('temp/tys/output.json','w') as f:
        json.dump(output,f)

def sam_masks_rgb():
    spath = 'data/temp/282/sam_masks'
    dpath = 'data/temp/282/sam_masks_rgb'
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
    from PIL import Image
    import cv2
    device = 'cuda'
    def get_entity(image):
        # from segment_anything import (SamAutomaticMaskGenerator, SamPredictor, sam_model_registry)
        # sam = sam_model_registry['vit_h'](checkpoint='./third_party/segment-anything/weights/sam_vit_h_4b8939.pth').to(device)
        # mask_generator = SamAutomaticMaskGenerator(
        #     model=sam,
        #     points_per_side=32,
        #     pred_iou_thresh=0.88,
        #     box_nms_thresh=0.7,
        #     stability_score_thresh=0.95,
        #     crop_n_layers=0,
        #     crop_n_points_downscale_factor=1,
        #     min_mask_region_area=100,
        # )
        from transformers import SamProcessor, SamModel
        from transformers import pipeline
        model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        generator =  pipeline("mask-generation", model='facebook/sam-vit-huge', device = device, points_per_batch = 2)
        with torch.no_grad():
            outputs = generator(image, points_per_batch = 2)
        inputs = processor(image, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
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
        masks = [record['segmentation'] for record in records]
        entity = [get_entity_image(np.array(image), record['segmentation']) for record in records]
        return torch.from_numpy(np.stack(entity)), torch.from_numpy(np.stack(masks))

    def get_semantics(entity):
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        inputs = clip_processor(images=entity, return_tensors='pt')
        inputs = inputs.to(clip_model.device)
        semantics = clip_model.get_image_features(**inputs)
        semantics = F.normalize(semantics,dim=-1).detach().cpu()
        return semantics
    
    def get_relevancy_map(entity, masks, semantics, ptexts, ntexts):
        from transformers import CLIPModel, CLIPProcessor
        def get_semantic_map(masks: torch.Tensor, semantics):
            semantic_map = torch.full(masks.shape[1:3], -1, dtype=torch.int64)
            semantics = torch.concat((semantics, torch.zeros((1, semantics.shape[-1]))), dim=0)
            for index, mask in enumerate(masks):
                semantic_map[mask] = index
            return semantics[semantic_map]
        def get_relevancy(raw_semantic_map: torch.Tensor, pembed: torch.Tensor, nembed: torch.Tensor):
            s = raw_semantic_map.shape[:-1]
            c = raw_semantic_map.shape[-1]
            raw_semantics = raw_semantic_map.flatten(0, -2)
            psim=pembed@raw_semantics.T # (p, i)
            nsim=nembed@raw_semantics.T # (n, i)
            nsim=nsim.unsqueeze(0).repeat_interleave(pembed.shape[0],dim=0) # (p, n ,i)
            psim=psim.unsqueeze(1).repeat_interleave(nembed.shape[0],dim=1) # (p, n, i)
            sim=torch.stack((psim,nsim), dim=-1) # (p, n, i, 2)
            sim=torch.softmax(10*sim, dim=-1) # (p, n, i, 2)
            sim, indice = sim[...,0].min(dim=1) # (p, i)
            return sim.unflatten(1, s)

        semantic_map = get_semantic_map(masks, semantics)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        pembed = clip_processor(text=ptexts, return_tensors='pt', padding=True)
        pembed = pembed.to(clip_model.device)
        pembed = clip_model.get_text_features(**pembed)
        pembed = F.normalize(pembed, dim=-1).detach().cpu()

        nembed = clip_processor(text=ntexts, return_tensors='pt', padding=True)
        nembed = nembed.to(clip_model.device)
        nembed = clip_model.get_text_features(**nembed)
        nembed = F.normalize(nembed, dim=-1).detach().cpu()

        relevancy_map = get_relevancy(semantic_map, pembed, nembed)
        return relevancy_map


    image = Image.open('data/temp/nanfeng/images/00000000001-00000001113-A01113.jpg')
    # entity, masks = get_entity(image)
    # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #     semantics = get_semantics(entity)
    # torch.save(entity, 'temp/entity.pth')
    # torch.save(masks, 'temp/masks.pth')
    # torch.save(semantics, 'temp/semantics.pth')

    entity = torch.load('temp/entity.pth')
    masks = torch.load('temp/masks.pth')
    semantics = torch.load('temp/semantics.pth')

    ptexts = ['house', 'building', 'pavilion', 'tree', 'vegetable field', 'car', 'pool', 'water', 'grass']
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        relevancy_map = get_relevancy_map(entity, masks, semantics, ptexts, ["object", "things", "stuff", "texture"])
    torch.save(relevancy_map, 'temp/relevancy_map.pth')

    relevancy_map = torch.load('temp/relevancy_map.pth')
    for ptext, map in zip(ptexts, relevancy_map):
        img=torch.from_numpy(np.array(image)).permute(2,0,1)
        img = (img*(map>0.5)).permute(1,2,0)
        Image.fromarray(img.numpy()).save(f'temp/{ptext}.jpg')

def test_dinov2():
    device = 'cuda'
    # inp = torch.randn(1, 3, 10, 12)
    # w = torch.randn(2, 3, 4, 5)
    # inp_unf = torch.nn.functional.unfold(inp, (4, 5))  # shape of inp_unf is (1,3*4*5,7*8)

    # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)  # shape of out_unf is (1,2,56)
    # #  以上代码相当于 inp_unf(1, 60, 56) .t() * w(2 , 3 * 4 * 5).t() → out_unf (1, 56, 2 ) → out_unf (1, 2, 56)

    # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))  # out.size() = (1,2,7,8)

    # print((torch.nn.functional.conv2d(inp, w) - out).abs().max())  # tensor(1.9073e-06)
    # F.conv2d()
    from transformers import AutoImageProcessor, AutoModel

    image = Image.open('data/temp/nanfeng/images/00000000001-00000001113-A01113.jpg')

    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs.to(model.device))
    last_hidden_states = outputs[0]

    print(outputs['last_hidden_state'].shape)

def convert_gs_to_splm(input: str, output: str):
    from plyfile import PlyData, PlyElement
    SH_C0 = 0.28209479177387814
    plydata = PlyData.read(input)
    elements: PlyElement = plydata.elements[0]
    data = elements.data
    vertex_element = np.empty((elements.count),dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('alpha', 'u1'),
    ])
    vertex_element[:] = list(zip(
        data['x'], 
        data['y'], 
        data['z'], 
        ((data['f_dc_0']*SH_C0+0.5).clip(0,1)*255).astype('u1'),
        ((data['f_dc_1']*SH_C0+0.5).clip(0,1)*255).astype('u1'),
        ((data['f_dc_2']*SH_C0+0.5).clip(0,1)*255).astype('u1'),
        np.ones((elements.count), 'u1'), strict=True))
    vertex_element = PlyElement.describe(vertex_element, 'vertex')
    PlyData([vertex_element]).write(output)

from umap import UMAP
def test_umap():
    from utils.visualization_utils import features_to_color
    f = torch.rand((4,5))
    print(features_to_color(f))

def test_qwenvl():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from transformers import modeling_utils
    if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
        modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-3B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Please first output bbox coordinates and names of every item in this image in JSON format, and then answer how many items are there in the image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) # parse message to ChatML format, type(text) = str
    image_inputs, video_inputs = process_vision_info(messages) # process images to list of PIL.Image
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ) # inputs['input_ids']：表示token的编码，token与text中的字符不是一一对应的，一般token比text中的字符多
    # inputs['attention_mask']：与inputs['input_ids']长度一致
    # inputs['pixel_values']：处理后的图像值，例如对于2048*1365的图会输出shape为(14308，1176)的pixel_values，主要是由于经过了缩放、切块patch、展平、动态分辨率处理，由H*W*C变成了patch 数量 * embedding 维度
    # inputs['image_grid_thw']：用来表示每张图像的[t, h, w]，t表示时间序列的编号，h表示patch-wise的高度，w表示patch-wise的宽度，使得模型可以支持时间序列的图像和不同的、分辨率的图像
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128) # 回答的token
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ] # 将输出中包含输入的内容的部分去除，因为是自回归模型所以输出会包含输入
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)


def cost_volume(left_feature, right_feature):
    feature_similarity = 'difference'
    max_disp = 192
    b, c, h, w = left_feature.size()

    if feature_similarity == 'difference':
        cost_volume = left_feature.new_zeros(b, c, max_disp, h, w)

        for i in range(max_disp):
            if i > 0:
                cost_volume[:, :, i, :, i:] = left_feature[:, :, :, i:] - right_feature[:, :, :, :-i]
            else:
                cost_volume[:, :, i, :, :] = left_feature - right_feature

    elif feature_similarity == 'concat':
        cost_volume = left_feature.new_zeros(b, 2 * c, max_disp, h, w)
        for i in range(max_disp):
            if i > 0:
                cost_volume[:, :, i, :, i:] = torch.cat((left_feature[:, :, :, i:], right_feature[:, :, :, :-i]),
                                                        dim=1)
            else:
                cost_volume[:, :, i, :, :] = torch.cat((left_feature, right_feature), dim=1)

    elif feature_similarity == 'correlation':
        cost_volume = left_feature.new_zeros(b, max_disp, h, w)

        for i in range(max_disp):
            if i > 0:
                cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                            right_feature[:, :, :, :-i]).mean(dim=1)
            else:
                cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)

    else:
        raise NotImplementedError

    cost_volume = cost_volume.contiguous()  # [B, C, D, H, W] or [B, D, H, W]

    return cost_volume

def get_lidar_depth():
    from glob import glob
    from utils.visualization_utils import scalar_to_color, save_image
    def read_dmb_file(file_path, is_confidence=False):
        import struct
        """
        读取DMB文件，基于depth.cpp和conf.cpp中的格式
        
        参数:
            file_path: DMB文件路径
            is_confidence: 是否为置信度文件（使用uint8_t而非float）
            
        返回:
            numpy数组形式的深度/置信度数据，如果失败返回None
        """
        try:
            with open(file_path, 'rb') as f:
                # 读取文件头信息
                type_val = struct.unpack('<i', f.read(4))[0]  # int32_t type
                h = struct.unpack('<i', f.read(4))[0]         # int32_t h (height)
                w = struct.unpack('<i', f.read(4))[0]         # int32_t w (width)
                nb = struct.unpack('<i', f.read(4))[0]        # int32_t nb (channels)
                
                # print(f"DMB文件信息: type={type_val}, 高度={h}, 宽度={w}, 通道数={nb}")
                
                # 检查类型是否正确
                if type_val != 1:
                    print(f"错误：不支持的DMB文件类型: {type_val}")
                    return None
                
                # 计算数据大小
                data_size = h * w * nb
                # 使用frombuffer加速解析
                if is_confidence:
                    raw = f.read(data_size)
                    data_array = np.frombuffer(raw, dtype=np.uint8, count=data_size)
                    data_array = data_array.reshape(h, w, nb).astype(np.float32)
                else:
                    raw = f.read(data_size * 4)
                    data_array = np.frombuffer(raw, dtype=np.float32, count=data_size)
                    data_array = data_array.reshape(h, w, nb)
                
                # 如果只有一个通道，去掉最后一维
                if nb == 1:
                    data_array = data_array.squeeze()
                
                return data_array
                
        except Exception as e:
            print(f"读取DMB文件失败 {file_path}: {e}")
            return None
    input_folder = 'data/temp/bangongshi/data'
    output_folder = 'data/temp/bangongshi/lidar_depth'
    os.makedirs(output_folder, exist_ok=True)
    depth_files = glob(f'{input_folder}/*smoothDepth.dmb')
    for depth_file in depth_files:
        depth_map = torch.from_numpy(read_dmb_file(depth_file).copy())[None, ...]
        depth_map = F.interpolate(
            depth_map.unsqueeze(0),
            size=(720, 960),  # (H, W)
            mode='bilinear',
        ).squeeze(0)
        save_image(scalar_to_color(depth_map.flatten()).reshape(720, 960, 3), f'{output_folder}/{os.path.basename(depth_file)}.jpg')

def main():
    # pth_to_json()
    # sam_masks_rgb()
    # pick_image()
    # test_clip()
    # test_dinov2()
    # convert_gs_to_splm('/home/moonite/code/SegAnyGAussians/data/temp/suzongbangongshi/output_models/point_cloud/iteration_30000/point_cloud.ply', '/home/moonite/code/SpatialLM/pcd/suzongbangongshi.ply')
    # convert_gs_to_splm('/home/moonite/code/SegAnyGAussians/data/temp/juweihui/output_models/point_cloud/iteration_30000/point_cloud.ply', '/home/moonite/code/SpatialLM/pcd/juweihui.ply')
    # convert_gs_to_splm('/home/moonite/code/SegAnyGAussians/data/temp/hualang/output_models/point_cloud/iteration_30000/point_cloud.ply', '/home/moonite/code/SpatialLM/pcd/hualang.ply')
    # test_umap()
    # test_qwenvl()
    # cost_volume(torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))
    get_lidar_depth()

if __name__ =='__main__':
    main()