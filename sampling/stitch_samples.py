# Script to stitch together several images to form a synthetic whole slide image made out of synthetic patches.
# This enables annotation of the patches more easily.
import sys

# Make patches of 512x512
# stitch them together 100x100 -> 512_000x512_000 img. If it doesn't fit, then make it smaller


#First: Sample 10k patches of size 512


from PIL import Image
import pandas as pd
from tqdm import tqdm
from aiosynimage.io.image_writer import ImageWriter
import numpy as np


def main():
    patch_size = 512
    side_length = 40  # Number of patches to stitch together, side length. So will be AxA patches total.
    write_png = False
    #subsample contains 160 imgs
    datadir = "/Users/Mees_1/MasterThesis/Aiosyn/data/generated_samples/512_glomeruli"
    #datadir = "/Users/Mees_1/MasterThesis/Aiosyn/data/rat-tissue" #Real tissue

    dfs = []
    image_paths = []

    for dataset_index in range(5):
        print("Dataset index", dataset_index)
        df = pd.read_csv(datadir + f"/patches_{dataset_index}.csv")
        dfs.append(df)
        for index, row in df.iterrows():
            image_paths.append(datadir + f"/generated_images_{dataset_index}/" + row['image_uuid'])


    images = [Image.open(path) for path in image_paths]
    images_np = [np.array(img) for img in images]

    print(f"{len(images)} images.")
    if write_png:
        print("Also saving the img as png ...")
        stitched_image = Image.new('RGB', (patch_size * side_length, patch_size * side_length))

    path_of_tif = "stitched_image_1.00_spacing.tif"
    shape = (side_length * patch_size, side_length * patch_size)
    # I have no idea what to put at spacing ... NOt sure if it even applies here or what it means
    writer = ImageWriter(path=path_of_tif, shape=shape, spacing=[1.0, 1.0], dtype='uint8', channels=3, tile_size=patch_size)

    for idx, img in tqdm(enumerate(images), total=side_length**2, desc="Stitching images together"):
        if idx >= side_length * side_length:
            break
        x = idx % side_length * patch_size
        y = idx // side_length * patch_size
        if write_png:
            stitched_image.paste(img, (x, y))
        writer.write(images_np[idx], x, y)

    writer.finalize()

    if write_png:
        print("Done. Saving ... ")
        stitched_image.save("stitched_image.png")
    print(f"Done. Created image of {side_length} x {side_length} patches of size {patch_size}")


if __name__ == "__main__":
    main()

