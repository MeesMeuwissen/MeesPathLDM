import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pandas as pd
from torchinfo import summary
from tqdm import tqdm
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torch.nn as nn
import zipfile
from aiosynawsmodules.services import s3
from aiosynawsmodules.services.sso import set_sso_profile
from aiosynawsmodules.services.s3 import download_directory


def load_model(path, device):
    model = torch.jit.load(path)
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    model.to(device=device)

    return model

def prepare_data(remote_path):
    print("Downloading the data from S3 ... ")
    if remote_path[-1] == '/':
        remote_path = remote_path[:-1]
    set_sso_profile(profile_name="aws-aiosyn-data", region_name="eu-west-1")

    dataset_name = remote_path.split("/")[-1]
    local_prefix = "/Users/Mees_1/MasterThesis/Aiosyn/data/generated_samples/"

    download_directory(remote_path, local_prefix + dataset_name)

    zippath = local_prefix + dataset_name + "/generated_images.zip"
    with zipfile.ZipFile(zippath, 'r') as zip_ref:
        zip_ref.extractall(local_prefix + dataset_name+"/generated_images")
    return local_prefix + dataset_name
def main():
    model_path = "../pretrained/kidney-hotel-1.pt" # locally only
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    s3_path = "s3://aiosyn-data-eu-west-1-bucket-ops/patch_datasets/generation/synthetic-data/05-16_1420-size=256/" #
    data_dir_path = "/Users/Mees_1/MasterThesis/Aiosyn/data/generated_samples/05-16_1420-size=256"
    # data_dir_path = None

    if data_dir_path == None:
        data_dir_path = prepare_data(s3_path)
        print(f"Data dir:", data_dir_path)
    df = pd.read_csv(data_dir_path + "/patches.csv")
    patches_dir = data_dir_path + "/generated_images"
    batch_size = 4 # or something else?

    subsample = 10
    # df = df[1:1+subsample]

    img_size = 572
    padding = 0 # 92 doesnt work, so we use 86 for analysis

    print("Loading model...")
    model = load_model(model_path, device)
    padder = nn.ReflectionPad2d(padding=padding) # Pad by 94 since aiosyn classifier removes 184 pixels
    # With an input size of 256x256, padding by 94 gives output of 260x260.


    total_rows = len(df) - 1  # Subtract 1 for the header

    occurences = 0
    # Loop through the DataFrame in batches
    for i in tqdm(range(0, total_rows, batch_size), desc="Evaluating"):
        batch = df.iloc[i + 1:i + batch_size + 1]  # Add 1 to start index to skip the header
        batched_uuids = batch["image_uuid"]
        batched_captions = list(batch["caption"])

        batched_images = []
        for uuid in batched_uuids:
            image = Image.open(patches_dir + "/" + uuid)
            #plt.imshow(image)
            #image.save(f"unmirrorred_{uuid}")
            #plt.show()
            padded_img = padder(pil_to_tensor(image) / 255)

            #plt.imshow(padded_img.permute(1,2,0))
            #to_pil_image(padded_img).save(f"mirrorred_{uuid}.png")
            #plt.show()
            batched_images.append(padded_img) # Convert to tensor, convert to float32, reflectionpadded

        assert len(batched_images) == len(batched_captions), f"There should be equal amounts of images and captions"

        batched_images = torch.stack(batched_images).to(device)
        predictions = model(batched_images)
        print(predictions.shape)
        max_channel_idx = torch.argmax(predictions, dim=1)

        # Convert mask to captions.
        mask_tensor = max_channel_idx.unsqueeze(1)
        captions_from_model = []
        for mask in mask_tensor.cpu():
            mask = mask.to(torch.uint8)
            mask_image = to_pil_image(mask)
            captions_from_model.append(get_caption(mask))
            #plt.imshow(mask_image)
            #mask_image.save(f"mask_{uuid}.png")
            #plt.show()

        batched_captions = list(batched_captions)


        for i in range(len(batched_captions)):
            # print(f"Caption used as prompt: {batched_captions[i]}")
            # print(f"Caption as result from model output: {captions_from_model[i]}")
            #
            # # print(f"Equal:" , batched_captions[i] == captions_from_model[i])
            #
            # print("===" * 10)
            if caption_criterium(batched_captions[i], captions_from_model[i]):
                # print("occurence detected ...")
                occurences += 1
    print(f"Occurence in {occurences} out of {len(df) - 1} imgs, which is {occurences / (len(df) - 1) * 100 :2f}%")
    print()

def get_caption(mask):
    pixels = np.array(mask).ravel()
    hist, bins = np.histogram(pixels, bins=np.arange(0, 9))  # 8 classes, ignore the class_0
    probabilities = hist / len(pixels)  # Probability of a random pixel being a certain class

    probabilities = np.append(probabilities, probabilities[4] + probabilities[5]) # Sum of glomeruli
    classes = ['White background, should be ignored', 'Arteries', 'Atrophic Tubuli', 'Tubuli', 'Glowmeruli',
               'Sclerotic Glomeruli', 'other kidney tissue', 'Dilated Tubuli', 'glomsum']
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

def caption_criterium(batched, model):
    if 'Glowmeruli' in model and 'Sclerotic' not in model:
        return True

    return False

if __name__ == '__main__':
    main()