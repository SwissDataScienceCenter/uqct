from pathlib import Path
from typing import Literal

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
    "val_range": (0.0, 234.59),
    "rotation_angle": 30,
}

KWARGS_LUNG = {
    "path": DATA_DIR / "lung/ground_truth_train",
    "rescale": 128,
    "val_range": (0.0, 1.0),
    "rotation_angle": 30,
}

KWARGS_COMPOSITE = {
    "path": DATA_DIR / "composite/SampleG-FBI22-Stitch-0-1-2.txm.nii",
    "rescale": 128,
    "im_size": 256,
    "val_range": (0.0, 10646.63),
    "file_range": [20, 360],
    "clip_range": [3e4, 5e4],
    "rotation_angle": 30,
}


def get_dataset(
    name: Literal["composite", "lamino", "lung"], high_res: bool = False
) -> tuple[Subset[torch.Tensor], Subset[torch.Tensor]]:
    settings = {
        "composite": {"kwargs": KWARGS_COMPOSITE, "filetype": "nii"},
        "lamino": {"kwargs": KWARGS_LAMINO, "filetype": "tiff"},
        "lung": {"kwargs": KWARGS_LUNG, "filetype": "h5"},
    }

    # We need 256x256 to mitigate 'inverse crime problem'
    if high_res:
        for v in settings.values():
            v["kwargs"]["rescale"] = 256

    dataset_type: Literal["nii", "tiff", "h5"] = settings[name]["filetype"]
    kwargs = settings[name]["kwargs"]

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
    train_set = Subset(dataset, perm[: round(0.95 * len(dataset))])  # type: ignore
    test_set = Subset(dataset, perm[round(0.95 * len(dataset)) :])  # type: ignore
    return train_set, test_set


if __name__ == "__main__":
    train_set, test_set = get_dataset("lung")
