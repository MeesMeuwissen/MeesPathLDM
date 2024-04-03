import io
from pathlib import Path

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST
import pandas
import random


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
        self.flip_p = config.get("flip_p", 0)  # Default to 0 (no flips)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(permute_channels),
            ]
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.csv.iloc[idx]["relative_path"].replace("{file}", "img")  # Read the img part
        img_path = os.path.join(self.data_dir, img_path)

        img = Image.open(img_path).convert("RGB")
        img = self.random_flips(img, self.flip_p)
        img = self.transform(img)

        if img.shape[1] > self.crop_size:
            img = self.get_random_crop(img, self.crop_size)

        caption = self.create_caption()
        # should be HWC
        assert img.shape == torch.Size([256, 256, 3]), "img shape should be [256,256,3] but is {}".format(img.shape)
        return {"image": img, "caption": caption}

    def create_caption(self):
        return "A PAS stained slide of a piece of kidney tissue"  # Generic caption

    def get_random_crop(self, image, crop_size):
        # Get the dimensions of the original image
        width, height = image.size

        # Calculate the maximum valid coordinates for the top-left corner of the crop
        max_x = width - crop_size
        max_y = height - crop_size

        # Generate random coordinates for the top-left corner of the crop
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Perform the crop
        cropped_image = image.crop((x, y, x + crop_size, y + crop_size))
        return cropped_image,

    def random_flips(self, img, p):
        if torch.rand(1) < p:
            img = F.hflip(img)
        if torch.rand(1) < p:
            img = F.vflip(img)
        return img


class KidneyConditional(KidneyUnconditional):
    def __init__(self, config=None):
        super(KidneyConditional, self).__init__(config)

    def __getitem__(self, idx):
        img_path = self.csv.iloc[idx]["relative_path"].replace("{file}", "img")  # Read the img part
        msk_path = self.csv.iloc[idx]["relative_path"].replace("{file}", "msk")  # Read the msk part

        img_path = os.path.join(self.data_dir, img_path)
        msk_path = os.path.join(self.data_dir, msk_path)
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path)

        img, msk = self.random_flips(img, msk, self.flip_p)
        img = self.transform(img)
        msk = self.transform(msk)

        if img.shape[1] > self.crop_size:
            img, msk = self.get_random_crop(img, msk, self.crop_size)

        caption = self.create_caption(msk)
        # should be HWC
        assert img.shape == torch.Size([256, 256, 3]), "img shape should be [256,256,3] but is {}".format(img.shape)
        return {"image": img, "caption": caption}

    def create_caption(self, mask):
        """
        The different classes:
        0: 0 # No mask (background)
        1: 1 # Arteries
        2: 8 # Arterioles
        3: 2 # Atrophic tubuli
        4: 8 # Background
        5: 8 # Capillaries
        6: 3 # Capsule
        7: 4 # Distal tubuli
        8: 8 # Fat
        9: 5 # Glomeruli
        10: 8 # Other
        11: 6 # Proximal tubuli
        12: 7 # Sclerotic glomeruli
        13: 0  # Undefined tubuli - don't learn from areas of undefined tubuli because it's not really background (it's tubuli)
        14: 8 # Vessels
        """
        pixels = np.array(mask).ravel()
        hist, bins = np.histogram(pixels, bins=np.arange(0, 16))  # 15 classes, some will be very rarely seen.
        percentages = hist / len(pixels) * 100

        rest = percentages[0] + percentages[4] + percentages[10] + percentages[13]
        caption = (
            "This image displays histological sections of renal tissue, showcasing various types of tissue structures. "
            "The composition of the image includes:\n"
            f"- Arteries: {percentages[1]:.2f}%\n"
            f"- Arterioles: {percentages[2]:.2f}%\n"
            f"- Atrophic tubuli: {percentages[3]:.2f}%\n"
            f"- Capillaries: {percentages[5]:.2f}%\n"
            f"- Capsule: {percentages[6]:.2f}%\n"
            f"- Distal tubuli: {percentages[7]:.2f}%\n"
            f"- Fat: {percentages[8]:.2f}%\n"
            f"- Glomeruli: {percentages[9]:.2f}%\n"
            f"- Proximal tubuli: {percentages[11]:.2f}%\n"
            f"- Sclerotic glomeruli: {percentages[12]:.2f}%\n"
            f"- Vessels: {percentages[14]:.2f}%\n"
            f"The other {rest:.2f}% of the image consists of background and undefined tubuli."
        )

        return caption

    def random_flips(self, img, msk, p):
        if torch.rand(1) < p:
            img = F.hflip(img)
            msk = F.hflip(msk)
        if torch.rand(1) < p:
            img = F.vflip(img)
            msk = F.vflip(msk)
        return img, msk

    def get_random_crop(self, image, mask, crop_size):
        # Get the dimensions of the original image
        width, height = image.shape[1], image.shape[0]

        # Calculate the maximum valid coordinates for the top-left corner of the crop
        max_x = width - crop_size
        max_y = height - crop_size

        # Generate random coordinates for the bottom-left corner of the crop
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Perform the crop
        cropped_image = image[y:y + crop_size, x:x + crop_size]
        cropped_mask = mask[y:y + crop_size, x:x + crop_size]
        return cropped_image, cropped_mask

class RatKidneyConditional(KidneyConditional):
    # Size of rat-tissue dataset: 109923 patches
    def create_caption(self, mask):
        """
        {'Arteries': 1},
        {'Atrophic tubuli': 2},
        {'Tubuli': 3},
        {'Glomeruli': 4},
        {'Sclerotic glomeruli': 5},
        {'Background': 6},
        {'Dilated tubuli': 7}]
        """
        pixels = np.array(mask).ravel()
        hist, bins = np.histogram(pixels, bins=np.arange(0, 9))  # 8 classes, ignore the class_0
        probabilities = hist / len(pixels) # Probability of a random pixel being a certain class

        classes = ['White background, should be ignored', 'Arteries', 'Atrophic Tubuli', 'Tubuli', 'Glomeruli', 'Sclerotic Glomeruli', 'other kidney tissue',
                   'Dilated Tubuli']
        thresholds = {'low': 0.2, 'medium': 0.4}  # Define thresholds for low, medium, and high prevalence

        caption = "This image showcases various types of tissue structures found in renal tissue. \n"
        for i, prob in enumerate(probabilities):
            if i == 0:
                continue #First value should be ignored
            if prob == 0:
                continue  # Skip classes with zero probability
            elif prob < thresholds['low']:
                caption += f"The image shows a low amount of {classes[i]}.\n"
            elif prob < thresholds['medium']:
                caption += f"The prevalence of {classes[i]} is medium.\n"
            else:
                caption += f"There is a lot of {classes[i]} visible in the image.\n"

        return caption


class OverfitOneBatch(RatKidneyConditional):

    def __getitem__(self, idx):

        idx = 20 # Always the same image, no matter idx

        img_path = self.csv.iloc[idx]["relative_path"].replace("{file}", "img")  # Read the img part
        msk_path = self.csv.iloc[idx]["relative_path"].replace("{file}", "msk")  # Read the msk part

        img_path = os.path.join(self.data_dir, img_path)
        msk_path = os.path.join(self.data_dir, msk_path)
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path)

        img = self.transform(img)

        if img.shape[1] > self.crop_size:
            img, msk = self.get_random_crop(img, msk, self.crop_size)

        caption = self.create_caption(msk)
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
