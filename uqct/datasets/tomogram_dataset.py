import os

import h5py
import torch

from uqct.datasets.base_dataset import BaseImageDataset


class H5Wrapper:
    def __init__(self, path):
        if os.path.isdir(path):
            folder = [
                filename
                for filename in os.listdir(path)
                if filename.endswith(".h5")
                or filename.endswith(".hdf5")
                or filename.endswith(".mat")
            ]
        else:
            folder = [os.path.basename(path)]
            path = os.path.dirname(path)

        self.path = path
        self.folder = folder
        idx_to_file_list = []
        global_index_to_local_list = []
        for i, filename in enumerate(folder):
            with h5py.File(os.path.join(path, filename), "r") as hr_data:
                for dataset_name in [
                    "images",
                    "tomogram_delta",
                    "data",
                    "mygroup/data",
                ]:
                    if dataset_name in hr_data:
                        break

                idx_to_file_list.append(
                    i * torch.ones(len(hr_data.get(dataset_name))).int()  # type: ignore
                )
                global_index_to_local_list.append(
                    torch.arange(len(hr_data.get(dataset_name)))  # type: ignore
                )

        self.idx_to_file = torch.cat(idx_to_file_list)
        self.global_index_to_local = torch.cat(global_index_to_local_list)

    def __getitem__(self, idx):
        file_id = self.idx_to_file[idx]
        local_index = self.global_index_to_local[idx]
        filename = self.folder[file_id]
        with h5py.File(os.path.join(self.path, filename), "r") as hr_data:
            for dataset_name in ["images", "tomogram_delta", "data", "mygroup/data"]:
                if dataset_name in hr_data:
                    break
            return hr_data.get(dataset_name)[local_index.item()]  # type: ignore

    def __len__(self):
        return len(self.idx_to_file)

    def __str__(self):
        folder_str = str(self.folder[:3])
        if len(self.folder) > 3:
            folder_str = folder_str[:-1] + ", ...]"
        obs_0 = self.__getitem__(0)
        obs_1 = self.__getitem__(1)
        min_0, max_0 = obs_0.min(), obs_0.max()  # type: ignore
        min_1, max_1 = obs_1.min(), obs_1.max()  # type: ignore
        return f"H5Wrapper(path={self.path}, folder={folder_str}, shape={self.shape}, obs_0: shape={obs_0.shape}, min={min_0}, max={max_0}, type={type(obs_0)}, obs_1: shape={obs_1.shape}, min={min_1}, max={max_1}, type={type(obs_1)})"  # type: ignore

    @property
    def shape(self):
        return (len(self), *self.__getitem__([0]).shape)  # type: ignore


class TomogramDataset(BaseImageDataset):
    def __init__(
        self,
        path,
        rescale=None,
        clip_range=None,
        val_range=None,
        rotation_angle=None,
    ):
        self.hr_tomogram = H5Wrapper(path)

        super().__init__(
            rescale=rescale,
            clip_range=clip_range,
            val_range=val_range,
            rotation_angle=rotation_angle,
        )

    def __getitem__(self, idx):  # type: ignore
        hr_image = torch.from_numpy(self.hr_tomogram[idx])
        if len(hr_image.shape) == 2:
            hr_image = hr_image.unsqueeze(0)

        hr_image = self.transform(hr_image)
        return hr_image.clip(0, 1)

    def __len__(self):  # type: ignore
        return len(self.hr_tomogram)

    def __str__(self) -> str:
        return f"TomogramDataset(hr_tomogram={self.hr_tomogram})"


if __name__ == "__main__":
    from pathlib import Path

    import lovely_tensors as lt

    lt.monkey_patch()

    DATA_DIR_CANDIDATES = (Path(x) for x in ("/mydata/chip/shared/data", "data"))

    while not (data_dir := next(DATA_DIR_CANDIDATES)).exists():
        print(f"Directory '{data_dir}' doesn't exist")
    print(f"Data directory: {str(data_dir)}")

    kwargs_lung = {
        "path": data_dir / "ground_truth_train",
        "rescale": 128,
        "train_transform": False,
        "rotation_angle": 30,
        "normalize_range": True,
    }

    dataset = TomogramDataset(**kwargs_lung)
    print(dataset)
    print(dataset[0])
