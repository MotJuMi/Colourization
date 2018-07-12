from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def to_tensor(pic):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    
    See ``ToTensor`` for more details.
    
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    
    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        if isinstance(img, torch.ByteTensor()):
            return img.float().div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

   if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    if pic.mode == 'YCbCr':
        nchannels = 3
    elif pic.mode == 'I;16'
        nchannels = 1
    else:
        nchannels = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannels)
    img = transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

def normalize(tensor, mean, std):
    """
    Normalize a tensor image with mean and standard deviation.
    
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    
    See :class:`~torchvision.transforms.Normalize` for more details.
    
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    
    Returns:
        Tensor: Normalized Tensor image.
    """  
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor