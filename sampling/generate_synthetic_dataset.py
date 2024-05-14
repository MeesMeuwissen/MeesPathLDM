import argparse
import glob
import os
import sys
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torchvision import transforms
from tqdm import tqdm
from ldm.data.text_cond.thesis_conditioning import RatKidneyConditional

from aiosynawsmodules.services.s3 import download_file, upload_file
from aiosynawsmodules.services.sso import set_sso_profile


def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def get_model(config_path, device, checkpoint):
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, checkpoint, device)
    return model

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l",
        "--location",
        type=str,
        const=True,
        default="maclocal",
        nargs="?",
        help="Running local, maclocal or remote",
    )
    parser.add_argument(
        "-c", "--config_path", type=str, const=True, default=False, nargs="?", help="Path to the config file"
    )
    parser.add_argument('-s', "--save_s3", type=bool, default=False, nargs="?", help="Save the images to S3?")
    parser.add_argument('-n', '--nr_samples', type=int, default=100, help="Number of synthetic images to generate.")

    return parser


def get_samples(model, shape, batch_size, caption_generator, opt):
    # scale of classifier free guidance
    scale = 1.5
    sampler = DDIMSampler(model)

    if opt.same_x_T:
        print("Using same starting noise (x_T).")
        x_T = torch.randn([batch_size] + shape, device=device)
    else:
        x_T = None

    def get_unconditional_token(batch_size):
        return [""] * batch_size

    def get_conditional_token(batch_size, caption):
        return [caption] * batch_size

    with torch.no_grad():
        # unconditional token for classifier free guidance
        ut = get_unconditional_token(batch_size)
        uc = model.get_learned_conditioning(ut).to(torch.float32)

        ct = get_conditional_token(batch_size, caption_generator.generate())
        cc = model.get_learned_conditioning(ct).to(torch.float32)

        print("Starting sampling ...")
        samples_ddim, inters = sampler.sample(
            opt.ddim_steps,
            batch_size,
            shape,
            cc,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=0,
            use_tqdm=True,
            x_T=x_T,
            log_every_t=opt.log_every_t
        )

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8).cpu()

    return x_samples_ddim

def save_sample(sample, output_dir):
    image_uuid = str(uuid.uuid4())
    image_path = os.path.join(output_dir, image_uuid + ".png")
    to_pil = transforms.ToPILImage()

    sample = to_pil(sample)
    sample.save(image_path)
    return image_path


def main(config, nr_samples, location, save_to_S3 = False):
    opt = config.sampling_stuff

    ckpt_path = opt.ckpt_path
    config.model.params.ckpt_path = ckpt_path
    # Model will be downloaded to /home/aiosyn/model.ckpt (see download_model())

    if location in ['remote']:
        print("Downloading model ...")
        download_model(ckpt_path)
        config.model.params.ckpt_path = "/home/aiosyn/model.ckpt"

    model = instantiate_from_config(config.model).to(device)
    model.eval()

    batch_size = opt.batch_size
    shape = [3, opt.size, opt.size]

    now = datetime.now()
    formatted_now = now.strftime("%m-%d_%H%M")
    if location in ['remote']:
        output_dir = f"/home/aiosyn/data/generated_samples/{formatted_now}_size={4*opt.size}"
    else:
        output_dir = f"/Users/Mees_1/MasterThesis/Aiosyn/data/generated_samples/{formatted_now}"
    os.makedirs(output_dir, exist_ok=True)

    caption_generator = instantiate_from_config(opt.caption_config)

    print(f"Generating {opt.batches} batches synthetic images of size {batch_size} ...")
    print(f"Saving to {output_dir} ... ")
    img_paths = []
    for i in tqdm(range(opt.batches)):
        print(f"Batch {i+1}/{opt.batches}...")
        samples = get_samples(model, shape, batch_size, caption_generator=caption_generator, opt=opt)
        for sample in samples:
            path = save_sample(sample, output_dir)
            img_paths.append(path)


    if opt.get("FID_path", False):
        print("Calculating FID...")
        fid = calculate_FID(paths=img_paths, FID_path=opt.FID_path,  device=device)
    else:
        fid = None
    # save some metadata to a file in output dir as well.

    with open(output_dir + "/metadata.txt", "w") as f:
        f.write(f"Model path used: {ckpt_path}\n")
        f.write(f"FID compared to real data: {fid}\n")
        f.write(f"Depth of sampling: {opt.ddim_steps}\n")
        f.write(f"Number of samples: {nr_samples}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Caption function used: {opt.caption_config.target.split('.')[-1]}\n")

    if save_to_S3 or location == "remote":
        print(f"Saving samples in {output_dir} to S3 ...")
        for i, img_path in enumerate(img_paths[:10]):
            upload_file(
                img_path,
                f"s3://aiosyn-data-eu-west-1-bucket-ops/patch_datasets/generation/synthetic-data/{formatted_now}-size={4 * opt.size}/subsample_{i}.png",
            )

        if opt.upload_all:
            print("Zipping and uploading all samples ... ")
            zip_directory(output_dir, "generated_images.zip")
            upload_file(
                "generated_images.zip",
                f"s3://aiosyn-data-eu-west-1-bucket-ops/patch_datasets/generation/synthetic-data/{formatted_now}-size={4*opt.size}/generated_images.zip",
            )
        upload_file(
            output_dir + "/metadata.txt",
            f"s3://aiosyn-data-eu-west-1-bucket-ops/patch_datasets/generation/synthetic-data/{formatted_now}-size={4*opt.size}/metadata.txt",
        )
    print("Done")


def zip_directory(directory, zip_filename):
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in glob.glob(os.path.join(directory, "*.png")):
            zipf.write(file, os.path.relpath(file, directory))


def calculate_FID(paths, FID_path, device):
    model = InceptionV3().to(device)
    mu_fake, sig_fake = calculate_activation_statistics(paths, model, device=device)
    with np.load(Path(f"/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/FID/FID_outputs/{FID_path}")) as f:
        m1, s1 = f["mu"], f["sig"]

    fid = calculate_frechet_distance(m1, s1, mu_fake, sig_fake)
    print(f"Calculated FID: {fid}")
    return fid


def add_taming_lib(loc):
    if loc in ["local", "maclocal"]:
        taming_dir = os.path.abspath("generationLDM/src/taming-transformers")
    elif loc == "remote":
        taming_dir = os.path.abspath("code/generationLDM/src/taming-transformers")
    else:
        assert False, "Unknown location"
    sys.path.append(taming_dir)


def download_model(path):
    # Running remotely, so model needs to be downloaded
    download_file(
        remote_path=path,
        local_path="/home/aiosyn/model.ckpt",
    )
    sys.path.append("/home/aiosyn/code")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser = get_parser()

    options, unknown = parser.parse_known_args()
    add_taming_lib(options.location)

    config_path = options.config_path
    save_to_S3 = options.save_s3
    nr_samples = options.nr_samples

    config = OmegaConf.load(config_path)
    main(config, nr_samples, options.location, save_to_S3)
