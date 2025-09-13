# import h5py
import torch

# from torch.utils.data import Dataset
# from torchvision.transforms.functional import resize, InterpolationMode
# from torchvision import transforms

# from chip.utils.fourier import fft_2D, ifft_2D
# from chip.utils import add_defects
# import torch.nn.functional as F
from PIL import Image
import math
import os
import numpy as np
# from chip.utils.utils import create_gaussian_filter
from uqct.datasets.base_dataset import BaseImageDataset

Image.MAX_IMAGE_PIXELS = None

class tiff_wrapper():
    def __init__(self, path, im_size=512):
        if os.path.isdir(path):
            folder = [filename for filename in os.listdir(path) if
                      filename.endswith('.tiff') or filename.endswith('.tif')]
        else:
            folder = [os.path.basename(path)]
            path = os.path.dirname(path)

        self.path = path
        self.folder = folder
        self.im_size = im_size
        self.images = {}
        sizes = []
        idx_to_file_list = []
        global_index_to_local_list = []
        coordinates_list = []

        for i, filename in enumerate(folder):
            image = Image.open(os.path.join(path, filename))
            w, h = image.size
            w_ = (w - im_size) / (math.ceil(w / im_size) - 1)
            h_ = (h - im_size) / (math.ceil(h / im_size) - 1)
            num_images = math.ceil(w / im_size) * math.ceil(h / im_size)
            row = torch.arange(num_images) // math.ceil(w / im_size)
            col = torch.arange(num_images) % math.ceil(w / im_size)
            x = col * w_
            y = row * h_

            coordinates_list.append(torch.stack([x, y], -1))
            idx_to_file_list.append(i * torch.ones(num_images).int())
            global_index_to_local_list.append(torch.arange(num_images))

        self.idx_to_file = torch.cat(idx_to_file_list)
        self.global_index_to_local = torch.cat(global_index_to_local_list)
        self.coordinates = torch.cat(coordinates_list)

    def __getitem__(self, idx):
        file_id = self.idx_to_file[idx]
        local_index = self.global_index_to_local[idx]
        coors = self.coordinates[idx].int().numpy()

        filename = self.folder[file_id]
        # image = Image.open(os.path.join(self.path, filename))
        image = self.get_image(filename)
        cropped_image = image.crop((coors[0], coors[1], coors[0] + self.im_size, coors[1] + self.im_size))
        return cropped_image
    
    def get_image(self, path):
        if not path in self.images:
            image = Image.open(os.path.join(self.path, path))
            self.images[path] = image
        return self.images[path]

    def __len__(self):
        return len(self.idx_to_file)

class TIFFDataset(BaseImageDataset):

    def __init__(self, path, im_size,
                 rescale=None, clip_range=None, normalize_range=False, rotation_angle=None,
                 contrast=None, train_transform=False, crop=None):
        super().__init__(path,
                 rescale=rescale, clip_range=clip_range, normalize_range=normalize_range, rotation_angle=rotation_angle,
                 contrast=contrast, train_transform=train_transform, crop=crop)
        self.im_size = im_size
        self.images = tiff_wrapper(path, im_size)

    def __getitem__(self, idx):
        image = torch.tensor(np.array(self.images[idx])).float()

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import lovely_tensors as lt
    import os

    lt.monkey_patch()

    DATA_PATH = '/mydata/chip/shared/data' if torch.cuda.is_available() else 'data'


    kwargs = {
        'path': 'data/DATASET_G7_170um_10nm_rect',
        'im_size':512,
        'gray_background': False,
        'train_transform': True,
        'to_gray': False,
        'rotation_angle': 30,
        'rescale': 512,
        'crop': (200, 200, 128),
    }

    trainSet = TIFFDataset(**kwargs)
    print(trainSet[0])
