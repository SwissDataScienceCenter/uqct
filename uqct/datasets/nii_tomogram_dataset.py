import torch
from PIL import Image
import math
import numpy as np
from uqct.datasets.base_dataset import BaseImageDataset
import nibabel as nib

Image.MAX_IMAGE_PIXELS = None

class nii_wrapper():
    def __init__(self, path, im_size=512, file_range=[0, -1]):
        self.path = path
        self.im_size = im_size

        self.all_images = nib.load(path).get_fdata()[file_range[0]:file_range[1]]
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
        cropped_image = image[coors[0]:coors[0] + self.im_size, coors[1]:coors[1] + self.im_size]
        return cropped_image

    def __len__(self):
        return len(self.idx_to_slice)

class NiiDataset(BaseImageDataset):

    def __init__(self, path, im_size,
                 rescale=None, clip_range=None, normalize_range=False, rotation_angle=None,
                 contrast=None, train_transform=False, crop=None, file_range=[0, -1]):
        super().__init__(path,
                 rescale=rescale, clip_range=clip_range, normalize_range=normalize_range, rotation_angle=rotation_angle,
                 contrast=contrast, train_transform=train_transform, crop=crop)
        self.im_size = im_size
        self.images = nii_wrapper(path, im_size, file_range)


    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).float()

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import lovely_tensors as lt
    import os
    from chip.utils.utils import create_circle_filter, create_gaussian_filter
    from chip.models.forward_models import fourier_filtering
    lt.monkey_patch()

    frequency_cut_out_radius = 30
    circle_filter = create_circle_filter(frequency_cut_out_radius, 512)
    gaussian_filter = create_gaussian_filter(sigma=15, size=512)

    current_filter = circle_filter

    kwargs = {
        'path': 'data/composite/SampleG-FBI22-Stitch-0-1-2.txm.nii',
        'im_size':512,
        'lr_forward_function': lambda x: fourier_filtering(x, current_filter),
        'gray_background': False,
        'train_transform': True,
        'to_gray': False,
        'rotation_angle': 30,
        'rescale': 512,
        'file_range': [20, 360],
        'clip_range': [3e4, 5e4]
    }

    trainSet = NiiDataset(**kwargs)
    print(trainSet[0])
