import h5py
import torch
import os

from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, InterpolationMode
from torchvision import transforms

from chip.utils.fourier import fft_2D, ifft_2D
from chip.utils import add_defects
import torch.nn.functional as F
from chip.utils.utils import create_gaussian_filter


class PairedTransform:
    """Applies the same transform to a pair of images."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2, seed=None):
        if seed is None:
            seed = torch.randint(0, 2 ** 32, ())
        torch.manual_seed(seed)
        img1 = self.transform(img1)
        torch.manual_seed(seed)
        img2 = self.transform(img2)
        return img1, img2


def contrast_transform(target, scale=10):
    cont_target = torch.sigmoid((target - 0.5) * scale)
    cont_target -= torch.min(cont_target)
    cont_target /= torch.max(cont_target)
    return cont_target


def add_gray_background(img, mask=None):
    if mask is None:
        w = img.shape[1]
        cp = torch.meshgrid(torch.arange(w, device=img.device), torch.arange(w, device=img.device))
        mask = (cp[0] - w / 2) ** 2 + (cp[1] - w / 2) ** 2 <= (w / 2) ** 2
    return img + 0.5 * mask * (1. - img)


def transform_to_gray(image):
    alpha = 0.15
    gaussian_filter = create_gaussian_filter(size=image.shape[-1], sigma=70)
    gaussian_filter /= torch.max(gaussian_filter)
    gray = alpha + (1 - alpha) * (image + 0.1)
    gray /= torch.max(gray)
    fft_gray = fft_2D(gray, ortho=True)
    fft_gray += 0.1 * torch.randn_like(fft_gray)
    gray = ifft_2D(fft_gray * gaussian_filter, ortho=True).real
    return gray


def to_synthetic(target):
    size = target.shape[-2:]
    target = F.interpolate(target.unsqueeze(0).unsqueeze(0), size=size, mode='bilinear', align_corners=True)[0, 0]
    target /= torch.max(target)
    cont_target = torch.sigmoid((target - 0.5) * 20)
    cont_target -= torch.min(cont_target)
    cont_target /= torch.max(cont_target)
    return cont_target

def get_circle(x, rescale):
    w = rescale if rescale is not None else x.shape[-1]
    cp = torch.cartesian_prod(torch.arange(w), torch.arange(w))
    return (cp[:, 0] - w / 2) ** 2 + (cp[:, 1] - w / 2) ** 2 <= (w / 2) ** 2

class BaseImageDataset(Dataset):
    def __init__(self, path,
                 rescale=None, clip_range=None, normalize_range=False, rotation_angle=None,
                 contrast=None, train_transform=False, crop=None,
                 to_synthetic=False, circle_crop=True):
        super().__init__()
        self.transforms = []
        if rotation_angle: self.add_rotation(angle=rotation_angle)
        if crop is not None: self.add_crop(*crop)
        if clip_range is not None: self.add_clip_range(*clip_range)
        if contrast: self.add_contrast(scale=contrast)
        if to_synthetic: self.add_to_synthetic()
        if train_transform: self.add_train_transform()
        if rescale: self.add_scale(width=rescale)
        if circle_crop:
            self.transforms.append(lambda x: x * get_circle(x, rescale).view(x.shape[-2:]))
        if normalize_range: self.add_normalize_range()

    def __getitem__(self, idx):
        pass
    @property
    def transform(self):
        return transforms.Compose(self.transforms)


    def add_rotation(self, angle=30):
        self.transforms.append(
            transforms.RandomAffine((angle, angle), (0, 0), (1., 1.), interpolation=InterpolationMode.BILINEAR))

    def add_to_synthetic(self):
        self.transforms.append(to_synthetic)

    def add_to_gray(self):
        self.transforms.append(transform_to_gray)

    def add_contrast(self, scale=10):
        def contrast(image):
            return contrast_transform(image, scale=scale)

        self.transforms.append(contrast)

    def add_scale(self, width=512):
        def scale(image):
            return resize(image, size=(width, width), antialias=True)

        self.transforms.append(scale)

    def add_normalize_range(self):
        def normalize_range(image):
            image -= image.min()
            image /= torch.max(image) + 1e-20
            return image

        self.transforms.append(normalize_range)

    def add_clip_range(self, minimum, maximum):
        def clip_range(image):
            image = torch.clip(image, minimum, maximum)
            return image - image.min()

        self.transforms.append(clip_range)

    def add_crop(self, xoffset, yoffset, width):
        def crop(image):
            return image[:, xoffset:xoffset + width, yoffset:yoffset + width]

        self.transforms.append(crop)

    def add_train_transform(self):
        self.transforms.append(
            transforms.RandomAffine((-180, 180), (0, 0), (1, 1.3), interpolation=InterpolationMode.BILINEAR))

    def __len__(self):
        pass