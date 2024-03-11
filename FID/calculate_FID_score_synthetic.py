import glob

from pathlib import Path
import os
import numpy as np
import torch
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

from aiosynawsmodules.services.s3 import download_directory, upload_file, download_file


def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def main():
    device = get_device()
    local_dir = "/home/aiosyn/data/synthetic"
    data_path = "s3://aiosyn-data-eu-west-1-bucket-ops/patch_datasets/generation/synthetic-data/03-08_1448-size=256"
    real_FID_path = "s3://aiosyn-data-eu-west-1-bucket-ops/patch_datasets/generation/real_activation_stats /FID_full.npz"  # Keep the space after stats !

    print("Downloading files ...")
    download_directory(
        remote_s3_url=data_path,
        local_dir=local_dir,
        overwrite=True,
        recursive=True,
        )
    download_file(real_FID_path, "/home/aiosyn/FID_real.npz")
    # Get list of all img paths

    img_paths = glob.glob(local_dir + "/*.png")
    try:
        print("5 elements in local_dir:", os.listdir(local_dir)[:5])
    except Exception:
        pass
    print(f"Calculating FID of {len(img_paths)} images ... ")

    model = InceptionV3().to(device)
    print("Calculating activation of data ...")
    mu_fake, sig_fake = calculate_activation_statistics(img_paths, model, device=device)
    print("Done")

    real_data = np.load(Path("/home/aiosyn/FID_real.npz"))
    mu_real = real_data["mu"]
    sig_real = real_data["sig"]
    print("Calculating FID ...")
    FID = calculate_frechet_distance(mu_real, sig_real, mu_fake, sig_fake)
    print("Done. FID = {:.2f}".format(FID))
    with open(f"/home/aiosyn/data/synthetic/metadata.txt", "a") as f:
        f.write(f"FID score when compared with real: {FID}\n")

    np.savez(f"/home/aiosyn/activation_statistics.npz", mu=mu_fake, sig=sig_fake)
    upload_file(
        f"/home/aiosyn/activation_statistics.npz", data_path + "activation_statistics.npz"
    )
    upload_file("/home/aiosyn/data/synthetic/metadata.txt", data_path + "metadata.txt", overwrite=True)
    print(f"Uploaded outputs to S3")


if __name__ == "__main__":
    main()

