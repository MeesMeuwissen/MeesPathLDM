import warnings
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import transforms
import os, sys
import pytorch_lightning as pl
from torchinfo import summary

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
from einops import rearrange
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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
    del config["model"]["params"]["first_stage_config"]["params"]["ckpt_path"]
    del config["model"]["params"]["unet_config"]["params"]["ckpt_path"]
    model = load_model_from_config(config, checkpoint, device)
    return model


if __name__ == "__main__":
    sys.path.append("src/taming-transformers/") #Add taming lib
    #ckpt_path = "logs/03-01T16-28_maclocal/checkpoints/notraining.ckpt"
    ckpt_path = "pretrained/srikar/epoch_3-001.ckpt" # The pretrained PLIP from srikar
    #ckpt_path = "pretrained/checkpoints/loaded_and_saved.pth"
    config_path = "pretrained/srikar/sampling_config/08-03T09-35-project.yaml"
    saved_ckpt_path = "pretrained/checkpoints/loaded_and_saved_lightning.ckpt"

    config = OmegaConf.load(config_path)
    #model = get_model(config_path, device, ckpt_path)
    model = instantiate_from_config(config.model)

    #m,u = model.load_state_dict(sd, strict=True)

    model.to(device)
    model.eval()
    # summary(model)
    # trainer = pl.Trainer()
    # trainer.model = model
    # trainer.save_checkpoint(saved_ckpt_path)
    # print("Saved checkpoint at {}".format(saved_ckpt_path))
    #
    # model = get_model(config_path, device, saved_ckpt_path)
    sampler1 = DDIMSampler(model)
    #sampler2 = DDIMSampler(model_reloaded)


    summary = "A H&E stained slide of a piece of kidney tissue" # Almost empty summary to simulate unconditional

    def get_unconditional_token(batch_size):
        return [""] * batch_size

    def get_conditional_token(batch_size, summary):
        # append tumor and TIL probability to the summary
        tumor = ["High tumor; low TIL;"] * (batch_size) #Keep this
        return [t + summary for t in tumor]

    batch_size = 4 #keep low, otherwise I run out of RAM :( Especially for larger imgs
    shape = [3, 64, 64] # shape of sampled imgs, CHW

    # scale of classifier free guidance
    scale = 1.5

    with torch.no_grad():
        # unconditional token for classifier free guidance
        ut = get_unconditional_token(batch_size)
        uc = model.get_learned_conditioning(ut).to(torch.float32)

        ct = get_conditional_token(batch_size, summary)
        cc = model.get_learned_conditioning(ct).to(torch.float32)

        print("Starting sampling for sampler 1 ...")
        samples_ddim, _ = sampler1.sample(
            5,
            batch_size,
            shape,
            cc,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=0,
            use_tqdm=True
        )
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8).cpu()

        grid = make_grid(x_samples_ddim, nrow=2)

        # to image
        grid = rearrange(grid, "c h w -> h w c").cpu().numpy()
        plt.imshow(grid, cmap="gray")
        plt.show()
        # print("Starting sampling for sampler 2 ...")
        # samples_ddim, _ = sampler2.sample(
        #     50,
        #     batch_size,
        #     shape,
        #     cc,
        #     verbose=False,
        #     unconditional_guidance_scale=scale,
        #     unconditional_conditioning=uc,
        #     eta=0,
        #     use_tqdm=True
        # )
        # x_samples_ddim = model.decode_first_stage(samples_ddim)
        # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        # x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8).cpu()
        #
        # grid = make_grid(x_samples_ddim, nrow=2)
        #
        # # to image
        # grid = rearrange(grid, "c h w -> h w c").cpu().numpy()
        # plt.imshow(grid, cmap="gray")
        #
        #
        # plt.show()
