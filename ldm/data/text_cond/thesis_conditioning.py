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
def permute_channels(x):
    return x.permute(1, 2, 0).contiguous()

class KidneyUnconditional(Dataset):
    def __init__(self, config=None):
        self.location = config.get("location")
        self.subsample = config.get("subsample")
        if self.location == "local":
            prefix = Path("/mnt/c/Users/MeesMeuwissen/Documents/Aiosyn/data/")
        elif self.location == "maclocal":
            prefix = Path("/Users/Mees_1/MasterThesis/Aiosyn/data/")
        elif self.location == "remote":
            prefix = Path("/home/aiosyn/data")
        else:
            raise ValueError("Wrong location. Please choose either 'local' or 'remote'.")

        if self.subsample:
            self.csv = prefix / config.get("csv").replace("patches.csv", "patches_subsample.csv")
            self.data_dir = prefix / Path(config.get("root").replace("patches", "patches_subsample"))

        else:
            self.csv = prefix / config.get("csv")
            self.data_dir = prefix / Path(config.get("root"))
        self.csv = pandas.read_csv(self.csv)

        self.slides_list = os.listdir(self.data_dir)
        self.size = min(config.get("size"), len(self.csv))

        self.crop_size = config.get("crop_size", None)
        self.flip_horizontal = config.get("flip_h", 0)  # Default to 0 (no flips)
        self.flip_vertical = config.get("flip_v", 0)


        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=self.flip_horizontal),
                transforms.RandomVerticalFlip(p=self.flip_vertical),
                transforms.ToTensor(),
                transforms.Lambda(permute_channels),
            ]
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        caption = "A PAS stained slide of a piece of kidney tissue"  # Generic caption

        img_path = self.csv.iloc[idx]["relative_path"].replace("{file}", "img")  # Read the img part
        img_path = os.path.join(self.data_dir, img_path)

        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)

        # should be HWC
        assert img.shape == torch.Size([256, 256, 3]), "img shape should be [256,256,3] but is {}".format(img.shape)
        return {"image": img, "caption": caption}


class KidneyConditional(Dataset):
    def __init__(self, config=None):
        self.location = config.get("location")
        self.data_dir = Path(config.get("root"))
        if self.location == "local":
            prefix = Path("/mnt/c/Users/MeesMeuwissen/Documents/Aiosyn/data/")
        elif self.location == "remote":
            prefix = Path("/tmp/data/")
        else:
            raise ValueError("Wrong location. Please choose either 'local' or 'remote'.")

        self.data_dir = prefix / self.data_dir
        self.csv = prefix / config.get("csv")
        self.csv = pandas.read_csv(self.csv)

        self.slides_list = os.listdir(self.data_dir)
        self.size = min(config.get("size"), len(self.csv))

        self.crop_size = config.get("crop_size", None)
        self.flip_horizontal = config.get("flip_h", 0)  # Default to 0 (no flips)
        self.flip_vertical = config.get("flip_v", 0)


        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=self.flip_horizontal),
                transforms.RandomVerticalFlip(p=self.flip_vertical),
                transforms.ToTensor(),
                transforms.Lambda(permute_channels),
            ]
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        caption = ""  # TODO: Fix this caption. It should ideally contain information from the slide, such as tissue type and distribution. Example: "H&E stain of a glomerulus, 40% X, 60% Y." These percentages could be obtained with use of Aiosyns models

        img_path = self.csv.iloc[idx]["relative_path"].replace("{file}", "img")  # Read the img part
        img_path = os.path.join(self.data_dir, img_path)

        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)

        # should be HWC
        assert img.shape == torch.Size([256, 256, 3]), "img shape should be [256,256,3] but is {}".format(img.shape)
        return {"image": img, "caption": caption}


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
