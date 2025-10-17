from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, resize


def get_circle(x, rescale):
    w = rescale if rescale is not None else x.shape[-1]
    cp = torch.cartesian_prod(torch.arange(w), torch.arange(w))
    return (cp[:, 0] - w / 2) ** 2 + (cp[:, 1] - w / 2) ** 2 <= (w / 2) ** 2


class BaseImageDataset(Dataset):
    def __init__(
        self,
        rescale=None,
        clip_range=None,
        val_range: tuple[float, float] | None = None,
        rotation_angle=None,
    ):
        super().__init__()
        self.transforms = []
        if rotation_angle:
            self.add_rotation(angle=rotation_angle)
        if clip_range is not None:
            self.add_clip_range(*clip_range)
        if rescale:
            self.add_scale(width=rescale)
        self.transforms.append(lambda x: x * get_circle(x, rescale).view(x.shape[-2:]))

        if val_range:
            self.add_normalize_range(val_range[0], val_range[1])

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor | np.ndarray:
        pass

    @property
    def transform(self):
        return transforms.Compose(self.transforms)

    def add_rotation(self, angle=30):
        self.transforms.append(
            transforms.RandomAffine(
                (angle, angle),
                (0, 0),
                (1.0, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            )
        )

    def add_scale(self, width=512):
        def scale(image):
            return resize(image, size=[width, width], antialias=True)

        self.transforms.append(scale)

    def add_normalize_range(self, min: float, max: float):
        def normalize_range(image):
            image -= min
            image /= max - min
            return image

        self.transforms.append(normalize_range)

    def add_clip_range(self, minimum, maximum):
        def clip_range(image):
            image = torch.clip(image, minimum, maximum)
            return image - image.min()

        self.transforms.append(clip_range)

    @abstractmethod
    def __len__(self) -> int:
        pass
