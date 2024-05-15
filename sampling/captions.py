import os
from pathlib import Path

import numpy as np
import pandas
from PIL import Image
import torchvision.transforms.functional as F


def unconditional():
    return "A PAS stained slide of a piece of kidney tissue"

def generate_normals(means, stddev):
    samples = []
    for m in means:
        samples.append(max(0, np.random.normal(m, stddev)))

    return samples


class CaptionGenerator():
    def generate(self):
        return ""
class RatKidneyConditional(CaptionGenerator):
    # Generate probabilities randomly if not given
    # The average probabilities in the rat-tissue dataset:
    # Readable:  [0.0037, 0.0001, 0.0271, 0.4418, 0.0121, 0.0219, 0.376, 0.1172]

    def generate(self, probabilities = None):
        if probabilities is None:
            averages = [3.72632114e-03, 1.39337593e-04, 2.70518503e-02, 4.41828884e-01, 1.21383491e-02, 2.18519807e-02,
                       3.76029002e-01, 1.17234275e-01]
            stddev = 0.1
            probabilities = generate_normals(means=averages, stddev=stddev)

        classes = ['White background, should be ignored', 'Arteries', 'Atrophic Tubuli', 'Tubuli', 'Glomeruli',
                   'Sclerotic Glomeruli', 'other kidney tissue',
                   'Dilated Tubuli']
        thresholds = {'low': 0.2, 'medium': 0.4}  # Define thresholds for low, medium, and high prevalence

        caption = "This image showcases various types of tissue structures found in renal tissue. \n"
        for i, prob in enumerate(probabilities):
            if i == 0:
                continue  # First value should be ignored
            if prob == 0:
                continue  # Skip classes with zero probability
            elif prob < thresholds['low']:
                caption += f"The image shows a low amount of {classes[i]}.\n"
            elif prob < thresholds['medium']:
                caption += f"The prevalence of {classes[i]} is medium.\n"
            else:
                caption += f"There is a lot of {classes[i]} visible in the image.\n"

        return caption


class Glomeruli(RatKidneyConditional):
    def generate(self):
        # Always returns the same caption, where tubuli are medium, glomeruli are high and the rest is 0
        return super().generate([0,0,0,0.3,1,0,0])

class CenteredGlomeruli(CaptionGenerator):

    def generate(self):
        caption = "This image showcases various types of tissue structures found in renal tissue. \n"
        caption += "The prevalence of Tubuli is medium.\n"
        caption += f"There is a lot of Glomeruli visible in the centre of the image.\n"
        return caption

class LowTubuliHighGlomeruli(CaptionGenerator):

    def generate(self):
        caption = "This image showcases various types of tissue structures found in renal tissue. \n"
        caption += "The image shows a low amount of Tubuli.\n"
        caption += f"There is a lot of Glomeruli visible in the centre of the image.\n"
        return caption


class RatKidneyLikeDataset(CaptionGenerator):
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
            print("Forgot to include location. Probably remote?")
            prefix = Path("/home/aiosyn/data")

        self.csv = prefix / config.get("csv")
        self.data_dir = prefix / Path(config.get("root"))

        self.csv = pandas.read_csv(self.csv)

        self.slides_list = os.listdir(self.data_dir)
        self.size = len(self.csv)
        self.id = 0

    def generate(self):

        msk_path = self.csv.iloc[self.id]["relative_path"].replace("{file}", "msk")  # Read the msk part
        self.id += 1
        if self.id > self.size:
            self.id = 0
            print("Resetting id")
        msk_path = os.path.join(self.data_dir, msk_path)
        msk = Image.open(msk_path)
        msk = F.pil_to_tensor(msk)

        caption = self.create_caption(msk)
        # should be HWC
        return caption

    def create_caption(self, mask):
        pixels = np.array(mask).ravel()
        hist, bins = np.histogram(pixels, bins=np.arange(0, 9))  # 8 classes, ignore the class_0
        probabilities = hist / len(pixels)  # Probability of a random pixel being a certain class

        classes = ['White background, should be ignored', 'Arteries', 'Atrophic Tubuli', 'Tubuli', 'Glomeruli',
                   'Sclerotic Glomeruli', 'other kidney tissue',
                   'Dilated Tubuli']
        thresholds = {'low': 0.2, 'medium': 0.4}  # Define thresholds for low, medium, and high prevalence

        caption = "This image showcases various types of tissue structures found in renal tissue. \n"
        for i, prob in enumerate(probabilities):
            if i == 0:
                continue  # First value should be ignored
            if prob == 0:
                continue  # Skip classes with zero probability
            elif prob < thresholds['low']:
                caption += f"The image shows a low amount of {classes[i]}.\n"
            elif prob < thresholds['medium']:
                caption += f"The prevalence of {classes[i]} is medium.\n"
            else:
                caption += f"There is a lot of {classes[i]} visible in the image.\n"

        return caption
