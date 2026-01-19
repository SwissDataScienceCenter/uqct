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
        "/cluster/home/mgaetzner/uq-xray-ct/data",
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
    "val_range": (0.0, 247.86),
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
    "file_range": [20, 360],
    "val_range": [3e4, 4e4],
    "rotation_angle": 30,
}

DatasetName = Literal["composite", "lamino", "lung"]


def get_dataset(
    name: DatasetName, high_res: bool = False
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

    dataset_type = settings[name]["filetype"]
    kwargs = settings[name]["kwargs"]

    dataset_class = TomogramDataset if dataset_type == "h5" else TIFFDataset
    dataset_class = NiiDataset if dataset_type == "nii" else dataset_class

    if dataset_type == "tiff" and "im_size" not in kwargs:
        kwargs["im_size"] = 512

    if dataset_type == "nii" and "clip_range" not in kwargs:
        kwargs["im_size"] = 512

    dataset = dataset_class(**kwargs)
    torch.manual_seed(0)
    perm = torch.randperm(len(dataset))
    with open(f"{name}_perm.txt", "w") as f:
        for idx in perm:
            f.write(f"{idx}\n")
    train_set = Subset(dataset, perm[: round(0.95 * len(dataset))])  # type: ignore
    test_set = Subset(dataset, perm[round(0.95 * len(dataset)) :])  # type: ignore
    return train_set, test_set  # type: ignore


if __name__ == "__main__":
    datasets = ("lamino", "lung")
    for ds_name in datasets:
        print(f"Dataset: {ds_name}")
        # print(f"Finding min and max pixel values in training and test set...")

        train_set, test_set = get_dataset(ds_name)
        # train_min = min(x.min().item() for x in train_set)
        # train_max = max(x.max().item() for x in train_set)
        # test_min = min(x.min().item() for x in test_set)
        # test_max = max(x.max().item() for x in test_set)

        # print(f"Train set: min={train_min}, max={train_max}")
        # print(f"Test set: min={test_min}, max={test_max}")
