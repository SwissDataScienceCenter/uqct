import h5py
import torch
import os


from uqct.datasets.base_dataset import BaseImageDataset


class h5_wrapper():
    def __init__(self, path):
        if os.path.isdir(path):
            folder = [filename for filename in os.listdir(path) if
                      filename.endswith('.h5') or filename.endswith('.hdf5') or filename.endswith('.mat')]
        else:
            folder = [os.path.basename(path)]
            path = os.path.dirname(path)

        self.path = path
        self.folder = folder
        sizes = []
        idx_to_file_list = []
        global_index_to_local_list = []
        for i, filename in enumerate(folder):
            with h5py.File(os.path.join(path, filename), 'r') as hr_data:
                for dataset_name in ['images', 'tomogram_delta', 'data', 'mygroup/data']:
                    if dataset_name in hr_data:
                        break
                idx_to_file_list.append(i * torch.ones(len(hr_data.get(dataset_name))).int())
                global_index_to_local_list.append(torch.arange(len(hr_data.get(dataset_name))))

        self.idx_to_file = torch.cat(idx_to_file_list)
        self.global_index_to_local = torch.cat(global_index_to_local_list)

    def __getitem__(self, idx):
        file_id = self.idx_to_file[idx]
        local_index = self.global_index_to_local[idx]
        filename = self.folder[file_id]
        with h5py.File(os.path.join(self.path, filename), 'r') as hr_data:
            for dataset_name in ['images', 'tomogram_delta', 'data', 'mygroup/data']:
                if dataset_name in hr_data:
                    break
            return hr_data.get(dataset_name)[local_index.item()]

    def __len__(self):
        return len(self.idx_to_file)
    @property
    def shape(self):
        return (len(self), *self.__getitem__([0]).shape)

class TomogramDataset(BaseImageDataset):
    def __init__(self, path, lr_path=None,
                 rescale=None, clip_range=None, normalize_range=False, rotation_angle=None,
                 contrast=None, train_transform=False, crop=None):
        self.hr_tomogram = h5_wrapper(path)
        self.lr_tomogram = None
        super().__init__(path, 
                 rescale=rescale, clip_range=clip_range, normalize_range=normalize_range, rotation_angle=rotation_angle,
                 contrast=contrast, train_transform=train_transform, crop=crop)

        if lr_path:
            self.lr_tomogram = h5_wrapper(lr_path)

    def __getitem__(self, idx):
        hr_image = torch.Tensor(self.hr_tomogram[idx])
        if len(hr_image.shape) == 2:
            hr_image = hr_image.unsqueeze(0)

        hr_image = self.transform(hr_image)
        return hr_image

    def __len__(self):
        return len(self.hr_tomogram)


if __name__ == '__main__':
    import lovely_tensors as lt
    import os

    lt.monkey_patch()

    DATA_PATH = '/mydata/chip/shared/data' if torch.cuda.is_available() else 'data'

    frequency_cut_out_radius = 30

    kwargs = {
        'path': os.path.join(DATA_PATH, 'p17299/tomogram_delta.mat'),
        'gray_background': False,
        'train_transform': False,
        'to_gray': False,
        'rotation_angle': 30,
        'rescale': 512
    }

    trainSet = TomogramDataset(**kwargs)
    print(trainSet[0])
