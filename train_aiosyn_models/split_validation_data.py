import os
import shutil
import sys

import pandas as pd
import csv


def main():
    df = pd.read_csv('/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/rat-glomeruli-gt/patches.csv')
    print(f"{len(df) = }")

    old_patches_root = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/rat-glomeruli-gt/patches/"

    val_csv_path = 'rat-glomeruli-true-validation/patches.csv'
    val_patches_root = 'rat-glomeruli-true-validation/patches/'

    train_100_0_csv_path = 'train_100_0/patches.csv'
    train_100_0_patches_root = 'train_100_0/patches'
    # Take the first 198 patches, and make it into a training set named train_100_0. The other patches are the true validation set

    train_df = df.iloc[:198]
    val_df = df.iloc[198:]

    for x, row in train_df.iterrows():
        # move the imgs and masks
        img_path = row["relative_path"].replace("{file}", "img")
        msk_path = row["relative_path"].replace("{file}", "msk")

        os.makedirs(train_100_0_patches_root + '/' + msk_path.split('/')[0], exist_ok=True)

        new_img_path = train_100_0_patches_root + '/' + img_path
        new_msk_path = train_100_0_patches_root + '/' + msk_path

        shutil.copy(old_patches_root + img_path, new_img_path)
        shutil.copy(old_patches_root + msk_path, new_msk_path)

    train_df.to_csv(train_100_0_csv_path)

    # Add the splits manually! Also zip manually!

    for x, row in val_df.iterrows():
        # move the imgs and masks
        img_path = row["relative_path"].replace("{file}", "img")
        msk_path = row["relative_path"].replace("{file}", "msk")

        os.makedirs(val_patches_root + '/' + msk_path.split('/')[0], exist_ok=True)

        new_img_path = val_patches_root + '/' + img_path
        new_msk_path = val_patches_root + '/' + msk_path

        shutil.copy(old_patches_root + img_path, new_img_path)
        shutil.copy(old_patches_root + msk_path, new_msk_path)

    val_df.to_csv(val_csv_path)

if __name__ == '__main__':
    main()
    print("Done")