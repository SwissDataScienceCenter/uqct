import math

import nibabel as nib
import torch
from PIL import Image

from uqct.datasets.base_dataset import BaseImageDataset

Image.MAX_IMAGE_PIXELS = None


class NIIWrapper:
    def __init__(self, path, im_size=512, file_range=[0, -1]):
        self.path = path
        self.im_size = im_size

        self.all_images = nib.load(path).get_fdata()[file_range[0] : file_range[1]]  # type: ignore
        idx_to_slice_list = []
        global_index_to_local_list = []
        coordinates_list = []

        for i, image in enumerate(self.all_images):
            w, h = image.shape
            w_ = (w - im_size) / (math.ceil(w / im_size) - 1)
            h_ = (h - im_size) / (math.ceil(h / im_size) - 1)
            num_images = math.ceil(w / im_size) * math.ceil(h / im_size)
            row = torch.arange(num_images) // math.ceil(w / im_size)
            col = torch.arange(num_images) % math.ceil(w / im_size)
            x = col * w_
            y = row * h_

            coordinates_list.append(torch.stack([x, y], -1))
            idx_to_slice_list.append(i * torch.ones(num_images).int())
            global_index_to_local_list.append(torch.arange(num_images))

        self.idx_to_slice = torch.cat(idx_to_slice_list)
        self.global_index_to_local = torch.cat(global_index_to_local_list)
        self.coordinates = torch.cat(coordinates_list)

    def __getitem__(self, idx):
        file_id = self.idx_to_slice[idx]
        coors = self.coordinates[idx].int().numpy()

        image = self.all_images[file_id]
        cropped_image = image[
            coors[0] : coors[0] + self.im_size, coors[1] : coors[1] + self.im_size
        ]
        return cropped_image

    def __len__(self):
        return len(self.idx_to_slice)


class NiiDataset(BaseImageDataset):

    def __init__(
        self,
        path,
        im_size,
        rescale=None,
        clip_range=None,
        val_range=None,
        rotation_angle=None,
        file_range=[0, -1],
    ):
        self.im_size = im_size
        self.images = NIIWrapper(path, im_size, file_range)
        super().__init__(
            rescale=rescale,
            clip_range=clip_range,
            val_range=val_range,
            rotation_angle=rotation_angle,
        )

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).float()

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    pass
