import sklearn.preprocessing
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def to_image(array, dataformat='HWC') -> Image.Image:
    if dataformat=='CHW':
        array=to_image_array(array, dataformat).transpose(1,2,0)
    else:
        array=to_image_array(array, dataformat)
    return Image.fromarray(array)

def to_image_array(array, dataformat='HWC') -> np.ndarray:
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
    array = array.reshape(-1,c)
    # pca = PCA(n_components=3)
    # array = pca.fit_transform(array)
    # array = sklearn.preprocessing.minmax_scale(array, (0,1))
    array = (array*255).astype(np.uint8)
    array = array.reshape(h,w,-1)

    if dataformat=='CHW':
        return array.transpose(2,0,1)
    else:
        return array

def save_image(array, path, dataformat='HWC'):
    image = to_image(array, dataformat)
    image.save(path)