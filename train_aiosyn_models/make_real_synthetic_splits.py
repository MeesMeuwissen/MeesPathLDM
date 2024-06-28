import os
import shutil
import sys
import random

import pandas as pd
import csv


def main():
    real_df = pd.read_csv('/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/train_100_0/patches.csv')
    synth_df = pd.read_csv('/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/synthetic-glomeruli/patches.csv')
    print(f"{len(real_df) = }, {len(synth_df) = }")

    real_ratio = 0.25
    new_patches_root = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/train_25_75/patches"
    new_patches_csv = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/train_25_75/patches.csv"

    real_patches_root = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/train_100_0/patches"
    synth_patches_root = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/synthetic-glomeruli"

    val_csv_path = 'rat-glomeruli-true-validation/patches.csv'
    val_patches_root = 'rat-glomeruli-true-validation/patches/'

    train_100_0_csv_path = 'train_100_0/patches.csv'
    train_100_0_patches_root = 'train_100_0/patches'
    # Take the first .75 indices from real, other .25 from synth

    split_index = int(real_ratio*198)
    print("Splitting the index at", split_index, f"which means {split_index} real patches, and {198 - split_index} validation patches.")

    real_indices = random.sample(range(198), split_index)
    synth_indices = random.sample(range(198), 198 - split_index)

    print(real_indices, synth_indices)

    with open(new_patches_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        for i, real_index in enumerate(real_indices):
            row = real_df.iloc[real_index]
            img_path = row["relative_path"].replace("{file}", "img")
            msk_path = row["relative_path"].replace("{file}", "msk")

            os.makedirs(new_patches_root + '/' + msk_path.split('/')[0], exist_ok=True)

            new_img_path = new_patches_root + '/' + img_path
            new_msk_path = new_patches_root + '/' + msk_path

            shutil.copy(train_100_0_patches_root + '/' + img_path, new_img_path)
            shutil.copy(train_100_0_patches_root + '/' + msk_path, new_msk_path)

            row = row.tolist()
            row[0] = i
            writer.writerow(row)

        print("Done with real")
        for i, synth_index in enumerate(synth_indices):
            row = synth_df.iloc[synth_index]
            img_path = row["relative_path"].replace("{file}", "img")
            msk_path = row["relative_path"].replace("{file}", "msk")

            os.makedirs(new_patches_root + '/' + msk_path.split('/')[0], exist_ok=True)

            new_img_path = new_patches_root + '/' + img_path
            new_msk_path = new_patches_root + '/' + msk_path

            shutil.copy(synth_patches_root + '/' + img_path, new_img_path)
            shutil.copy(synth_patches_root + '/' + msk_path, new_msk_path)

            row = [i + split_index] + row.tolist()
            writer.writerow(row)

    # shuffle the csv

    df = pd.read_csv(new_patches_csv)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(new_patches_csv, index=False)


if __name__ == '__main__':
    main()
    print("Done")