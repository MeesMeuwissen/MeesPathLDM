import argparse
import os
import sys
import uuid
from datetime import datetime

import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from aiosynawsmodules.services.s3 import upload_directory, download_file
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
    return parser


def get_samples(model, shape, batch_size, depth_of_sampling, summary, tumor_desc):
    scale = 1.5  # Scale of classifier free guidance

    sampler = DDIMSampler(model)

    def get_unconditional_token(batch_size):
        return [""] * batch_size

    def get_conditional_token(batch_size, summary):
        # append tumor and TIL probability to the summary
        tumor = [tumor_desc] * (batch_size)  # Keep this
        return [t + summary for t in tumor]

    with torch.no_grad():
        # unconditional token for classifier free guidance
        ut = get_unconditional_token(batch_size)
        uc = model.get_learned_conditioning(ut).to(torch.float32)

        ct = get_conditional_token(batch_size, summary)
        cc = model.get_learned_conditioning(ct).to(torch.float32)

        samples_ddim, _ = sampler.sample(
            50,
            batch_size,
            shape,
            cc,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=0,
            use_tqdm=False,
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


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    config_path = "generationLDM/configs/sampling/sampling.yaml"
    ckpt_path = "generationLDM/pretrained/srikar/epoch_3-001.ckpt"
    save_to_s3 = True

    size = 128
    summary = "A H&E stained slide of a piece of kidney tissue"
    tumor_desc = "High tumor; low TIL;"  # What to do with this??

    nr_of_samples = 10  # Nr of samples to generate
    depth_of_sampling = 50  # Steps in the sampling process
    batch_size = 4 # 256 with batch size 4 crashes aws (out of memory)
    shape = [3, size, size]

    now = datetime.now()
    formatted_now = now.strftime("%m-%d_%H%M")
    if opt.location == 'maclocal':
        output_dir = f"/Users/Mees_1/MasterThesis/Aiosyn/data/generated_samples/{formatted_now}_size={size}"
    elif opt.location == 'remote':
        output_dir =f"/home/aiosyn/data/generated_samples/{formatted_now}_size={size}"
    os.makedirs(output_dir, exist_ok=True)
    try:
        model = get_model(config_path, device, ckpt_path)
    except FileNotFoundError:
        model = get_model('code/' + config_path, device, 'code/' + ckpt_path)

    print(f"Generating {nr_of_samples} synthetic images of size {size} ...")
    print(f"Saving to {output_dir} ... ")
    for i in tqdm(range(nr_of_samples // batch_size + 1)):
        samples = get_samples(model, shape, batch_size, depth_of_sampling, summary, tumor_desc)
        for sample in samples:
            save_sample(sample, output_dir)

    if save_to_s3 or opt.location == 'remote':
        print(f"Saving samples in {output_dir} to S3 ...")
        #set_sso_profile("aws-aiosyn-data", region_name="eu-west-1")
        upload_directory(
            output_dir,
            f"s3://aiosyn-data-eu-west-1-bucket-ops/patch_datasets/generation/synthetic-data/{formatted_now}-size={size}/",
        )



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
        local_path="/home/aiosyn/code/generationLDM/pretrained/srikar/epoch_3-001.ckpt",
    )
    sys.path.append("/home/aiosyn/code")


if __name__ == "__main__":
    parser = get_parser()

    opt, unknown = parser.parse_known_args()
    add_taming_lib(opt.location)
    if opt.location == "remote":
        # Model will be downloaded to pretrained/srikar/epoch_3-001.ckpt (see download_model())
        print("Downloading model ...")
        model_path = "s3://aiosyn-data-eu-west-1-bucket-ops/models/generation/unet/pathldm/epoch_3-001.ckpt"
        download_model(model_path)

    main()
