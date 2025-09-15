from pathlib import Path

import torch
from torch.utils.data import Subset

from uqct.datasets.nii_tomogram_dataset import NiiDataset
from uqct.datasets.tiff_tomogram_dataset import TIFFDataset
from uqct.datasets.tomogram_dataset import TomogramDataset

DATA_DIR_CANDIDATES = [
    Path(x)
    for x in (
        "/mydata/chip/shared/data",
        "../data",
        "./data",
        "/cluster/scratch/mgaetzner/data",
    )
]
DATA_DIR = None
for x in DATA_DIR_CANDIDATES:
    if x.is_dir():
        DATA_DIR = x
if DATA_DIR is None:
    raise FileNotFoundError(
        f"Couldn't find data directory. Checked {DATA_DIR_CANDIDATES}"
    )

KWARGS_LAMINO = {
    "path": DATA_DIR / "lamino_tiff",
    "rescale": 128,
    "im_size": 256,
    "train_transform": False,
    "rotation_angle": 30,
    "normalize_range": False,
    "normalize_range_global": True,
}

KWARGS_LUNG = {
    "path": DATA_DIR / "ground_truth_train",
    "rescale": 128,
    # 'im_size': 256,
    "train_transform": False,
    "rotation_angle": 30,
    "normalize_range": False,
    "normalize_range_global": True,
}

KWARGS_COMPOSITE = {
    "path": DATA_DIR / "composite/SampleG-FBI22-Stitch-0-1-2.txm.nii",
    "rescale": 128,
    "im_size": 256,
    "range": (0.0, 10646.63),
    "train_transform": False,
    "file_range": [20, 360],
    "clip_range": [3e4, 5e4],
    "rotation_angle": 30,
    "normalize_range": False,
    "normalize_range_global": True,
}


def get_dataset(kwargs: dict, dataset_type: str) -> tuple[Subset, Subset]:
    dataset_class = TomogramDataset if dataset_type == "h5" else TIFFDataset
    dataset_class = NiiDataset if dataset_type == "nii" else dataset_class

    if dataset_type == "tiff" and "im_size" not in kwargs:
        kwargs["im_size"] = 512

    if dataset_type == "nii" and "clip_range" not in kwargs:
        kwargs["im_size"] = 512
        kwargs["clip_range"] = [3e4, 5e4]

    dataset = dataset_class(**kwargs)
    torch.manual_seed(0)
    perm = torch.randperm(len(dataset))
    trainSet = Subset(dataset, perm[: round(0.95 * len(dataset))])  # type: ignore
    testSet = Subset(dataset, perm[round(0.95 * len(dataset)) :])  # type: ignore
    return trainSet, testSet
