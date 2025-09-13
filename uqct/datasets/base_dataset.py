import torch
import torch.nn.functional as F
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
        path,
        rescale=None,
        clip_range=None,
        normalize_range=False,
        rotation_angle=None,
        contrast=None,
        train_transform=False,
        crop=None,
        to_synthetic=False,
        circle_crop=True,
    ):
        super().__init__()
        self.transforms = []
        if rotation_angle:
            self.add_rotation(angle=rotation_angle)
        if crop is not None:
            self.add_crop(*crop)
        if clip_range is not None:
            self.add_clip_range(*clip_range)
        if contrast:
            self.add_contrast(scale=contrast)
        if to_synthetic:
            self.add_to_synthetic()
        if train_transform:
            self.add_train_transform()
        if rescale:
            self.add_scale(width=rescale)
        if circle_crop:
            self.transforms.append(
                lambda x: x * get_circle(x, rescale).view(x.shape[-2:])
            )
        if normalize_range:
            self.add_normalize_range()

    def __getitem__(self, idx):
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
            return image[:, xoffset : xoffset + width, yoffset : yoffset + width]

        self.transforms.append(crop)

    def add_train_transform(self):
        self.transforms.append(
            transforms.RandomAffine(
                (-180, 180), (0, 0), (1, 1.3), interpolation=InterpolationMode.BILINEAR
            )
        )

    def __len__(self):
        pass

