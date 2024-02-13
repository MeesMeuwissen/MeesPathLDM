import io
from pathlib import Path

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST
import pandas


class TCGADataset(Dataset):
    """Dataset with tumor presence labels in text"""

    def __init__(self, config=None):
        split = config.get("split")
        data_dir = Path(config.get("root"))
        self.crop_size = config.get("crop_size", None)

        num_levels = config.get("num_levels", 2)
        assert num_levels in [2, 3, 4], "num_levels must be 2 or 3 or 4"
        self.p_uncond = config.get("p_uncond", 0)

        # Low, high if two levels else low, mid, high
        self.levels = ["low", "high"] if num_levels == 2 else ["low", "mid", "high"]

        # Load .h5 dataset
        self.data_file = h5py.File(data_dir / "TCGA_BRCA_10x_448_tumor.hdf5", "r")

        # Load metadata
        arr1 = np.load(data_dir / f"train_test_brca_tumor/{split}.npz", allow_pickle=True)
        self.indices = arr1["indices"]
        self.summaries = arr1["summaries"].tolist()
        self.prob_tumor = arr1["prob_tumor"].tolist()
        self.prob_til = arr1["prob_til"].tolist()

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def get_random_crop(img, size):
        x = np.random.randint(0, img.shape[1] - size)
        y = np.random.randint(0, img.shape[0] - size)
        img = img[y : y + size, x : x + size]
        return img

    def __getitem__(self, idx):
        x_idx = self.indices[idx]

        tile = self.data_file["X"][x_idx]
        tile = np.array(Image.open(io.BytesIO(tile)))

        image = (tile / 127.5 - 1.0).astype(np.float32)
        if self.crop_size:
            image = self.get_random_crop(image, self.crop_size)

        # Random horizontal and vertical flips
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()

        wsi_name = self.data_file["wsi"][x_idx].decode()
        folder_name = self.data_file["folder_name"][x_idx].decode()
        text_prompt = self.summaries[wsi_name]

        # Convert tumor infiltrating lymphocytes to levels low / mid / high and add to text prompt
        p_til = self.prob_til.get(wsi_name, {}).get(folder_name)
        if p_til is not None:
            p_til = int(p_til * len(self.levels))
            text_prompt = f"{self.levels[p_til]} til; {text_prompt}"

        # Convert tumor presence to levels low / mid / high and add to text prompt
        p_tumor = self.prob_tumor.get(wsi_name, {}).get(folder_name)
        if p_tumor is not None:
            p_tumor = int(p_tumor * len(self.levels))
            text_prompt = f"{self.levels[p_tumor]} tumor; {text_prompt}"

        # Replace text prompt with unconditional text prompt with probability p_uncond
        # Dont replace if p_til is positive
        if np.random.rand() < self.p_uncond and (p_til is None or p_til == 0):
            text_prompt = ""

        return {
            "image": image,
            "caption": text_prompt,
        }


class HandwrittenDigits(Dataset):
    def __init__(self, config=None):
        super().__init__()
        self.cmap = config.get("cmap")
        self.size = config.get("size")
        split = config.get("split")
        if split == "train":
            trainsplit = True
        elif split == "test":
            trainsplit = False
        else:
            raise ValueError(f"Dataset split: {split} not supported. Choose from ['train', 'test'].")
        self.mnist_dataset = MNIST(root="./MNISTdata", train=trainsplit)

        indices = list(range(0, self.size))
        self.mnist_dataset = Subset(self.mnist_dataset, indices)

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                lambda x: x.convert("RGB"),
                transforms.ToTensor(),
                lambda x: x.permute(1, 2, 0),
            ]
        )

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, id):
        # In this loop, id are simply integers.
        # transform the image to tensor
        img = self.transform(self.mnist_dataset[id][0])
        label = self.mnist_dataset[id][1]

        # Include own created caption
        caption = f"A handwritten number {label}."

        # HWC should be [256,256,3]
        assert img.shape == torch.Size([256, 256, 3]), "img shape should be [256,256,3] but is {}".format(img.shape)

        return {"image": img, "caption": caption}


class KidneyUnconditional(Dataset):
    def __init__(self, config=None):
        self.data_dir = Path(config.get("root"))
        self.slides_list = os.listdir(self.data_dir)
        self.csv = pandas.read_csv(config.get("csv"))
        self.size = config.get("size", None)

        self.crop_size = config.get("crop_size", None)
        self.flip_horizontal = config.get("flip_h", 0)  # Default to 0 (no flips)
        self.flip_vertical = config.get("flip_v", 0)

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=self.flip_horizontal),
                transforms.RandomVerticalFlip(p=self.flip_vertical),
                transforms.ToTensor(),
                lambda x: x.permute(1, 2, 0),
            ]
        )

    def __len__(self):
        if self.size is not None:
            return self.size
        return len(self.csv)

    def __getitem__(self, idx):
        caption = ""  # empty caption to simulate unconditional training ?

        img_path = self.csv.iloc[idx]["relative_path"].replace("{file}", "img")  # Read the img part
        img_path = os.path.join(self.data_dir, img_path)

        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)

        # should be HWC
        assert img.shape == torch.Size([256, 256, 3]), "img shape should be [256,256,3] but is {}".format(img.shape)
        return {"image": img, "caption": caption}
