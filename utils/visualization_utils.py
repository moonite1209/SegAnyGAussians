import sklearn.preprocessing
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

def to_image(array, dataformat='HWC') -> Image.Image:
    if dataformat=='CHW':
        array=to_image_array(array, dataformat).transpose(1,2,0)
    else:
        array=to_image_array(array, dataformat)
    return Image.fromarray(array)

def to_image_array(array, dataformat='HWC', use_pca=False) -> np.ndarray:
    if isinstance(array, np.ndarray):
        pass
    elif isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    else:
        raise TypeError()
    if dataformat=='CHW':
        array = array.transpose(1,2,0)
    else:
        pass

    h,w,c = array.shape
    array = (array*255).astype(np.uint8)

    if dataformat=='CHW':
        return array.transpose(2,0,1)
    else:
        return array

def save_image(array, path, dataformat='HWC'):
    image = to_image(array, dataformat)
    image.save(path)

def feature_map_to_image(feature_map: np.ndarray, dataformat='HWC')->np.ndarray:
    array = feature_map
    if isinstance(feature_map, np.ndarray):
        pass
    elif isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()
    else:
        raise TypeError()
    if dataformat=='CHW':
        feature_map = feature_map.transpose(1,2,0)
    elif dataformat=='HW':
        feature_map = feature_map[...,None]
    else:
        pass
    
    h,w,c = feature_map.shape
    if c==1:
        feature_map_norm = cv2.normalize(feature_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        feature_map_rgb = cv2.applyColorMap(feature_map_norm, cv2.COLORMAP_JET)
    else:
        if c==2:
            feature_map = np.concat((feature_map, np.zeros((h,w,1))), axis=-1)
        feature_map_flat = feature_map.reshape(-1,c)
        pca = PCA(n_components=3)
        feature_map_pca = pca.fit_transform(feature_map_flat).reshape(h, w, -1)
        feature_map_rgb = cv2.normalize(feature_map_pca, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    if dataformat=='CHW':
        return feature_map_rgb.transpose(2,0,1)
    else:
        return feature_map_rgb