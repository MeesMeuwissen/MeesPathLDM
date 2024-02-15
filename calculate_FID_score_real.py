from pathlib import Path
import pandas
from pytorch_fid.fid_score import calculate_activation_statistics
from pytorch_fid.inception import InceptionV3




# Need to get a list of paths to the datafiles...

prefix = Path("/mnt/c/Users/MeesMeuwissen/Documents/Aiosyn/datatest/first_patch_dataset_1.00_spacing/patches_subsample")
csv = pandas.read_csv("/mnt/c/Users/MeesMeuwissen/Documents/Aiosyn/datatest/first_patch_dataset_1.00_spacing/patches_subsample.csv")
img_paths = []

for i in range(0,len(csv)):
    img_path = csv.iloc[i]["relative_path"].replace("{file}", "img")  # Read the img part
    img_paths.append(img_path)

print(len(img_paths))
print("Done!")