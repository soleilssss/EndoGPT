from PIL import Image
import scipy.io
import numpy as np

def load_semseg(filename, loader_type):
    if loader_type == 'PIL':
        semseg = np.array(Image.open(filename), dtype=np.int)
    elif loader_type == 'gray':
        semseg = np.asarray(Image.open(filename)
                                             .convert('1')).astype(np.uint8)
    elif loader_type == 'gray_multiclass':
        semseg = np.asarray(Image.open(filename)
                                             .convert('L')).astype(np.uint8)
    elif loader_type == 'MAT':
        semseg = scipy.io.loadmat(filename)['LabelMap']
    return semseg