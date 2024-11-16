import pathlib
import h5py
import random
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Literal


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = PIL.Image.open(path)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)
    return img


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = PIL.Image.open(path)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg = np.stack([seg == 0, seg == 128, seg == 255])
    seg = seg.astype(np.float32)
    return seg


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in sorted(path.glob("*.bmp")):
        img = process_img(file, size=size)
        seg_file = file.with_suffix(".png")
        seg = process_seg(seg_file, size=size)
        data.append((img / 255.0, seg))
    return data


class WBCDataset(Dataset):
    def __init__(
        self,
        dataset: Literal["JTSC", "CV"] = "JTSC",
        support_frac: float = 0.3,
        support_size: int = 5,
        image_size: Tuple[int, int] = (128, 128),
        label: Optional[Literal["nucleus", "cytoplasm", "background"]] = None,
    ):
        # Define dataset path based on the dataset name
        dataset_path = pathlib.Path("WBCDatasets") / {"JTSC": "Dataset 1", "CV": "Dataset 2"}[dataset]
        self.dataset = dataset
        self.support_frac = support_frac
        self.support_size = support_size
        self.image_size = image_size
        self.label = label
        self._data = []
        self.support_idxs = []
        self.main_idxs = []

        # Load data from the folder
        T = torch.from_numpy
        self._data = [
            (T(x)[None], T(y)) for x, y in load_folder(dataset_path, size=self.image_size)
        ]

        if self.label is not None:
            self._ilabel = {"cytoplasm": 1, "nucleus": 2, "background": 0}[self.label]

        # Create support and main indices
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        support_end = int(np.floor(self.support_frac * N))
        self.support_idxs = p[:support_end]
        self.main_idxs = p[support_end:]

    @classmethod
    def load_from_hdf5(cls, hdf5_path: str, support_size: int):
        with h5py.File(hdf5_path, 'r') as f:
            # Load metadata
            dataset = f.attrs['dataset']
            support_frac = f.attrs['support_frac']
            image_size = tuple(f.attrs['image_size'])
            label = f.attrs['label']

            # Load indices
            support_idxs = f["support_idxs"][:]
            main_idxs = f["main_idxs"][:]

            # Load each image and mask from HDF5
            data = [
                (torch.tensor(f[f"img_{i}"][:]), torch.tensor(f[f"seg_{i}"][:]))
                for i in range(len(support_idxs) + len(main_idxs))
            ]

        # Define dataset path based on the dataset name
        dataset_path = pathlib.Path("WBCDatasets") / {"JTSC": "Dataset 1", "CV": "Dataset 2"}[dataset]

        # Create an instance of the dataset
        instance = cls(
            dataset=dataset,
            support_frac=support_frac,
            support_size=support_size,
            image_size=image_size,
            label=label,
        )
        instance._data = data
        instance.support_idxs = support_idxs
        instance.main_idxs = main_idxs
        print(f"Dataset loaded from HDF5 at {hdf5_path} with support_size={support_size}")
        return instance

    def __len__(self):
        return len(self.main_idxs)

    def __getitem__(self, idx):
        # Get the target image and label
        main_idx = self.main_idxs[idx]
        target_img, target_label = self._data[main_idx]

        if self.label is not None:
            target_label = target_label[self._ilabel][None]

        # Sample support images and labels
        support_indices = random.sample(list(self.support_idxs), self.support_size)
        support_imgs = []
        support_labels = []
        for s_idx in support_indices:
            s_img, s_label = self._data[s_idx]
            if self.label is not None:
                s_label = s_label[self._ilabel][None]
            support_imgs.append(s_img)
            support_labels.append(s_label)

        # Stack support images and labels
        support_imgs = torch.stack(support_imgs)  # Shape: (support_size, 1, H, W)
        support_labels = torch.stack(support_labels)  # Shape: (support_size, 1, H, W)

        return target_img, support_imgs, support_labels, target_label

    def save_hdf5(self, path: str):
        with h5py.File(path, 'w') as f:
            # Save metadata
            f.attrs['dataset'] = self.dataset  # Save dataset name
            f.attrs['support_frac'] = self.support_frac
            f.attrs['support_size'] = self.support_size
            f.attrs['image_size'] = self.image_size
            f.attrs['label'] = self.label

            # Save indices
            f.create_dataset("support_idxs", data=self.support_idxs)
            f.create_dataset("main_idxs", data=self.main_idxs)

            # Save image and mask data
            for i, (img, seg) in enumerate(self._data):
                f.create_dataset(f"img_{i}", data=img.numpy(), compression="gzip")
                f.create_dataset(f"seg_{i}", data=seg.numpy(), compression="gzip")

        print(f"Dataset saved as HDF5 at {path}")
