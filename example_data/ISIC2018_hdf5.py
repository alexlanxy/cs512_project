import pathlib
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import h5py


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = PIL.Image.open(path).convert("L")
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255
    return img.copy()


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    color_to_label = {
        (0, 0, 0, 255): 0,  # Background class
        (255, 255, 255, 255): 1  # Target class
    }

    seg = PIL.Image.open(path).convert("RGBA")
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    label_mask = np.zeros(seg.shape[:2], dtype=np.int8)

    for color, label in color_to_label.items():
        mask = np.all(seg == color, axis=-1)
        label_mask[mask] = label

    label_mask = label_mask[None]
    return label_mask.copy()


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    image_path = path / "Image"
    mask_path = path / "Mask"
    print("Absolute path:", mask_path.resolve())

    img_files = sorted(image_path.glob("*.jpg"))

    for img_file in tqdm(img_files, desc="Processing images and masks"):
        img = process_img(img_file, size=size)

        base_name = img_file.stem.split("_")[1]
        mask_file = mask_path / f"ISIC_{base_name}_segmentation.png"

        if mask_file.exists():
            seg = process_seg(mask_file, size=size)
            data.append((img, seg))
        else:
            print(f"No corresponding mask found for {img_file.name}")

    return data


class ISICDataset(Dataset):
    def __init__(self, label: int, support_size: int = 5, image_size: Tuple[int, int] = (128, 128), load_from_hdf5: bool = False):
        self.label = label
        self.support_size = support_size
        self.image_size = image_size

        if not load_from_hdf5:
            path = pathlib.Path('ISIC2018_Train')
            print("Absolute path:", path.resolve())
            self._data = load_folder(path, size=self.image_size)

            rng = np.random.default_rng(42)
            N = len(self._data)
            p = rng.permutation(N)

            # Define 60% for support and the rest (40%) for train+val
            support_end = int(0.6 * N)
            self.support_idxs = p[:support_end]
            self.main_idxs = p[support_end:]  # Combine train and val indices
        else:
            # Leave everything uninitialized for now; will be loaded by load_hdf5
            self._data = []
            self.support_idxs = []
            self.main_idxs = []

    def __len__(self):
        return len(self.main_idxs)

    def __getitem__(self, idx):
        target_idx = self.main_idxs[idx]
        target_img, target_label = self._data[target_idx]

        # Sample support images
        support_indices = random.sample(list(self.support_idxs), self.support_size)
        support_imgs = [self._data[s_idx][0] for s_idx in support_indices]
        support_labels = [self._data[s_idx][1] for s_idx in support_indices]

        # Convert to tensors
        T = torch.from_numpy
        target_img = T(target_img)[None]  # Shape: (1, H, W)
        target_label = T((target_label == self.label).astype(np.float32))  # Shape: (1, H, W)

        support_imgs = torch.stack([T(img)[None] for img in support_imgs])  # Shape: (support_size, 1, H, W)
        support_labels = torch.stack(
            [T((seg == self.label).astype(np.float32)) for seg in support_labels])  # Shape: (support_size, 1, H, W)

        return target_img, support_imgs, support_labels, target_label

    def save_hdf5(self, path: str):
        with h5py.File(path, 'w') as f:
            # Save metadata
            f.attrs['label'] = self.label
            f.attrs['support_size'] = self.support_size
            f.attrs['image_size'] = self.image_size

            # Save indices
            f.create_dataset("support_idxs", data=self.support_idxs)
            f.create_dataset("main_idxs", data=self.main_idxs)

            # Save image and mask data
            for i, (img, seg) in enumerate(self._data):
                f.create_dataset(f"img_{i}", data=img, compression="gzip")
                f.create_dataset(f"seg_{i}", data=seg, compression="gzip")

        print(f"Dataset saved as HDF5 at {path}")

    @classmethod
    def load_hdf5(cls, path: str, support_size: int = None):
        with h5py.File(path, 'r') as f:
            # Load metadata
            label = f.attrs['label']
            saved_support_size = f.attrs['support_size']
            image_size = tuple(f.attrs['image_size'])

            # Override support_size if specified
            if support_size is None:
                support_size = saved_support_size

            support_idxs = f["support_idxs"][:]
            main_idxs = f["main_idxs"][:]

            # Load each image and mask from HDF5
            loaded_data = [(f[f"img_{i}"][:], f[f"seg_{i}"][:]) for i in range(len(support_idxs) + len(main_idxs))]

        # Create an instance of ISICDataset and assign loaded data
        instance = cls(label=label, support_size=support_size, image_size=image_size, load_from_hdf5=True)
        instance._data = loaded_data
        instance.support_idxs = support_idxs
        instance.main_idxs = main_idxs
        print(f"Dataset loaded from HDF5 at {path} with support_size={support_size}")
        return instance