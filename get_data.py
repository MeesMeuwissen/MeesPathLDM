import logging
import os
import zipfile
from pathlib import Path

from aiosynawsmodules.services import s3
from aiosynawsmodules.services.sso import set_sso_profile


def download_dataset_from_s3(
    dataset_name: str, data_bucket_root: str, save_location: str, subsample: bool = False
) -> None:
    """Download data from S3 for use during training, validation or testing

    Args:
        dataset_name: The name of the dataset to be downloaded (usually "patches")
        data_bucket_root: The source of the data to be downloaded (e.g. "s3://aiosyn-data-eu-west-1-bucket-ops/patch_datasets/")
        save_location: The location (folder) where the downloaded data will be stored
        subsample: Whether to download the subsample instead of the whole thing (useful for debugging purposes)
    """

    # Ensure the directory where the data will be saved exists
    os.makedirs(save_location, exist_ok=True)

    if subsample is True:
        dataset_name = dataset_name + "_subsample"

    # The data itself has a name (dataset_name) and an extension (we need both the .zip and the .csv)
    remote_data_path = os.path.join(data_bucket_root, dataset_name + ".zip")
    local_data_path = os.path.join(save_location, dataset_name + ".zip")

    remote_csv_path = os.path.join(data_bucket_root, dataset_name + ".csv")
    local_csv_path = os.path.join(save_location, dataset_name + ".csv")

    if not s3.exists_s3(remote_data_path):
        logging.warning(f"{remote_data_path} S3 file does not exist in the cloud, skipping...")
    else:
        # download partition to local
        from aiosynawsmodules.services.s3 import download_file

        print(f"Downloading {dataset_name + '.csv'} from S3 ...")
        print(f"Downloading {dataset_name + '.zip'} from S3 ... ")

        # download the patch data.
        download_file(remote_path=remote_data_path, local_path=local_data_path, overwrite=True)
        download_file(remote_path=remote_csv_path, local_path=local_csv_path, overwrite=True)

        print(f"Downloads completed, extracting ...")
        logging.info("Downloaded {remote} to {local}".format(remote=remote_data_path, local=local_data_path))

        # Extract all the data from the .zip-file:
        os.makedirs(os.path.join(save_location, dataset_name), exist_ok=True)
        with zipfile.ZipFile(file=os.path.join(save_location, dataset_name + ".zip"), mode="r") as f:
            f.extractall(os.path.join(save_location, dataset_name))
        logging.info("Unzipped file {local}".format(local=local_data_path))
        print("Unzipped files")


# s3://aiosyn-data-eu-west-1-bucket-ops/patch_datasets/

def download_dataset(dataset_name:str, location:str = "local", subsample:bool = True) -> None:
    #Example dataset_name: "first_patch_dataset_1.00_spacing/patches"

    set_sso_profile(profile_name="aws-aiosyn-workloads-dev", region_name="eu-west-1")

    assert location in [
        "local",
        "remote",
        "maclocal",
    ], "Location must be either 'local' or 'remote'. Local means on your laptop, remote on aws."

    if location == "local":
        save_location = "/mnt/c/Users/MeesMeuwissen/Documents/Aiosyn/data"
    elif location == "maclocal":
        save_location = "/Users/Mees_1/Master Thesis/Aiosyn/data"
    elif location == "remote":
        save_location = "/tmp/data"

    print("Location:", location, "\nSave location:", save_location)
    download_dataset_from_s3(
        dataset_name=dataset_name,
        data_bucket_root="s3://aiosyn-data-eu-west-1-bucket-ops/patch_datasets/",
        save_location=save_location,
        subsample=subsample,
    )

if __name__ == "__main__":
    download_dataset("first_patch_dataset_1.00_spacing/patches", "maclocal", False)