from pathlib import Path
import pandas
import numpy as np
import os
import torch
from pytorch_fid.fid_score import calculate_activation_statistics
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import save_fid_stats

if __name__ == "__main__":
    subsample = False

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if subsample:
        prefix = Path("/Users/Mees_1/Master Thesis/Aiosyn/data/first_patch_dataset_1.00_spacing/patches_subsample")
        csv = pandas.read_csv(
            "/Users/Mees_1/Master Thesis/Aiosyn/data/first_patch_dataset_1.00_spacing/patches_subsample.csv"
        )

        file_path_output = os.getcwd() / Path("FID/FID_outputs/FID_subsample.npz")

    else:
        prefix = Path("/Users/Mees_1/Master Thesis/Aiosyn/data/first_patch_dataset_1.00_spacing/patches")
        csv = pandas.read_csv("/Users/Mees_1/Master Thesis/Aiosyn/data/first_patch_dataset_1.00_spacing/patches.csv")

        file_path_output = os.getcwd() / Path("FID/FID_outputs/FID_full.npz")

    img_paths = []

    # Get a list of all img paths ...
    for i in range(0, len(csv)):
        img_path = csv.iloc[i]["relative_path"].replace("{file}", "img")  # Read the img part
        img_paths.append(prefix / img_path)

    print(f"{len(img_paths) = } ")

    model = InceptionV3().to(device)

    print("Calculating activation of data ...")
    mu, sig = calculate_activation_statistics(img_paths, model, device=device)
    print("Done")

    print(f"{mu.shape = } {sig.shape = }")

    # Define the directory path
    directory_path = os.getcwd() / Path("FID/FID_outputs")
    # Create the directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    np.savez(file_path_output, mu=mu, sig=sig)
    print(f"Saved files to {directory_path}")

    print("Loading once more ...")
    data = np.load(file_path_output)

    # Access individual arrays by their keys
    mu_loaded = data['mu']
    sig_loaded = data['sig']

    print("Array 1:")
    print(mu_loaded)

    print("Array 2:")
    print(sig_loaded)

    print(f"{np.all(mu_loaded == mu) = }, {np.all(sig_loaded == sig) = }")

