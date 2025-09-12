import random

import odl
import torch
from chip.utils.utils import apply_circle_mask
from torch.utils.data import Dataset
import numpy as np


def random_shapes(interior=False):
    if interior:
        x_0 = 1.4*np.random.rand() - 0.70
        y_0 = 1.4*np.random.rand() - 0.70
    else:
        x_0 = 2 * np.random.rand() - 1.0
        y_0 = 2 * np.random.rand() - 1.0

    return (0.1 + np.random.rand() * 0.3,
            np.random.rand() * 0.3, np.random.rand() * 0.2,
            x_0, y_0,
            np.random.rand() * 2 * np.pi)

def random_shapes_small(interior=False):
    if interior:
        x_0 = 1.4*np.random.rand() - 0.70
        y_0 = 1.4*np.random.rand() - 0.70
    else:
        x_0 = 2 * np.random.rand() - 1.0
        y_0 = 2 * np.random.rand() - 1.0

    return (np.random.rand() *0.2 + 0.8,
            np.random.rand() * 0.1, 0.002 + np.random.rand() * 0.01,
            x_0, y_0,
            np.random.rand() * 2 * np.pi)


def random_phantom(spc, n_ellipse=50, interior=False, form='ellipse'):
    n = np.random.poisson(n_ellipse)
    shapes = np.array(
        [random_shapes(interior=interior) for _ in range(n // 5)] +
        [random_shapes_small(interior=interior) for _ in range(4 * n // 5)]
    )
    if form == 'ellipse':
        return odl.phantom.ellipsoid_phantom(spc, shapes)
    if form == 'rectangle':
        return odl.phantom.cuboid(spc)
    else:
        raise Exception('unknown form')


class EllipsesDataset(Dataset):
    def __init__(self, image_size=(1024, 1024), num_samples=1000):
        self.image_size = image_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        space = odl.uniform_discr([100, 100], [924, 924], self.image_size,
                                  dtype='float32')

        phantom = random_phantom(space, n_ellipse=random.randint(100, 200), interior=True, form='ellipse')
        t = torch.tensor(phantom.tensor).unsqueeze(0).clip(0, 1)
        return t, apply_circle_mask(t), idx