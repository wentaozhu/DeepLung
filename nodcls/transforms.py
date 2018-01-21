from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
from torch.autograd import Variable
torch.cuda.set_device(0)
def resample3d(inp,inp_space,out_space=(1,1,1)):
    # Infer new shape
    # inp = torch.from_numpy(inp)
    # inp=torch.FloatTensor(inp)
    # inp=Variable(inp)
    inp = inp.cuda()
    out = resample1d(inp,inp_space[2],out_space[2]).permute(0,2,1)
    out = resample1d(out,inp_space[1],out_space[1]).permute(2,1,0)
    out = resample1d(out,inp_space[0],out_space[0]).permute(2,0,1)
    return out

def resample1d(inp,inp_space,out_space=1):
    #Output shape
    print inp.size(), inp_space, out_space
    out_shape = list(np.int64(inp.size()[:-1]))+[int(np.floor(inp.size()[-1]*inp_space/out_space))] #Optional for if we expect a float_tensor
    out_shape = [int(item) for item in out_shape]
    # Get output coordinates, deltas, and t (chord distances)
    # torch.cuda.set_device(inp.get_device())
    # Output coordinates in real space
    coords = torch.cuda.HalfTensor(range(out_shape[-1]))*out_space
    delta = coords.fmod(inp_space).div(inp_space).repeat(out_shape[0],out_shape[1],1)
    t = torch.cuda.HalfTensor(4,out_shape[0],out_shape[1],out_shape[2]).zero_()
    t[0] = 1
    t[1] = delta
    t[2] = delta**2
    t[3] = delta**3    
    # Nearest neighbours indices
    nn = coords.div(inp_space).floor().long()    
    # Stack the nearest neighbors into P, the Points Array
    P = torch.cuda.HalfTensor(4,out_shape[0],out_shape[1],out_shape[2]).zero_()
    for i in range(-1,3):
        P[i+1] = inp.index_select(2,torch.clamp(nn+i,0,inp.size()[-1]-1))    
    #Take catmull-rom  spline interpolation:
    return 0.5*t.mul(torch.cuda.HalfTensor([[ 0,  2,  0,  0],
                            [-1,  0,  1,  0],
                            [ 2, -5,  4, -1],
                            [ -1, 3, -3,  1]]).mm(P.view(4,-1))\
                                                              .view(4,
                                                                    out_shape[0],
                                                                    out_shape[1],
                                                                    out_shape[2]))\
                                                              .sum(0)\
                                                              .squeeze()

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            # print(t)
            img = t(img)
        return img

class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        # for t, m, s in zip(tensor, self.mean, self.std):

        tensor.sub_(self.mean).div_(self.std)
        return tensor
from scipy.ndimage.interpolation import zoom
class RandomScale(object):
    ''' Randomly scale from scale size list '''
    def __init__(self, size, interpolation=Image.BILINEAR):
        # assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 3)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        # scale = np.random.permutation(len(self.size))[0] / 32.0
        scale = random.randint(self.size[0], self.size[-1]+1) #(self.size[np.random.permutation(len(self.size))[0]])#, \
                 # self.size[np.random.permutation(len(self.size))[0]], \
                 # self.size[np.random.permutation(len(self.size))[0]])
        # print img.shape, scale, img.shape*scale
        # print('scale', 32.0/scale)
        return zoom(img, (scale, scale, scale), mode='nearest')#resample3d(img,(32,32,32),out_space=scale)#zoom(img, scale) #img.resize(scale, self.interpolation) resample3d(img,img.shape,out_space=scale)
class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 3)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h, d = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

class ZeroOut(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        self.size = int(size)

    def __call__(self, img):
        w,h,d = img.shape #size
        x1 = random.randint(0, w-self.size) #np.random.permutation(w-self.size)[0]
        y1 = random.randint(0, h-self.size) # np.random.permutation(h-self.size)[0]
        z1 = random.randint(0, d-self.size) # np.random.permutation(d-self.size)[0]
        img1 = np.array(img)
        # print 'zero out', x1, y1, z1, w, h, d, self.size
        img1[x1:x1+self.size, y1:y1+self.size, z1:z1+self.size] = np.array(np.zeros((self.size, self.size, self.size)))
        return np.array(img1)
class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            pic = np.expand_dims(pic, -1)
            # print('before tensor', pic.shape)
            img = torch.from_numpy(pic.transpose((3, 0, 1, 2)))
            # backward compatibility
            return img.float()#.div(255)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()#.div(255)
        else:
            return img
class CenterCrop(object):
    """Crops the given PIL.Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class Pad(object):
    """Pad the given PIL.Image on all sides with the given "pad" value.

    Args:
        padding (int or sequence): Padding on each border. If a sequence of
            length 4, it is used to pad left, top, right and bottom borders respectively.
        fill: Pixel fill value. Default is 0.
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be padded.

        Returns:
            PIL.Image: Padded image.
        """
        return ImageOps.expand(img, border=self.padding, fill=self.fill)


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = int(size)
        self.padding = int(padding)

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            # print 'scale out', img.shape
            pad = int(self.padding/2)
            img1 = np.ones((img.shape[0]+pad, img.shape[1]+pad, img.shape[2]+pad))*170
            bg = int(self.padding/2)
            img1[bg:bg+img.shape[0], bg:bg+img.shape[1], bg:bg+img.shape[2]] = np.array(img)
            img = np.array(img1)
            # img = ImageOps.expand(img, border=self.padding, fill=170)

        w, h, d = img.shape#size
        th, tw, td = self.size
        # print 'pad out', w, h, d, th, tw, td
        if w == tw and h == th and d == td:
            return img
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        z1 = random.randint(0, d - td)
        return np.array(img[x1:x1+th, y1:y1+tw, z1:z1+td])
        # return img.crop((x1, y1, x1 + tw, y1 + th, z1 + td))

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return np.array(img[:, :, ::-1]) #.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomZFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return np.array(img[::-1, :, :])
        return img

class RandomYFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return np.array(img[:, ::-1, :])
        return img

class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))
