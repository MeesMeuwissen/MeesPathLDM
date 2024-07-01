import warnings

import torch
from omegaconf import OmegaConf
import sys

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from einops import rearrange


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def show_images(images, title=""):
    num_images = len(images)

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))

    # Create an empty axis for the title
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.title(title, fontsize=16)  # Adjust fontsize as needed

    for i in range(num_images):
        image = images[i].permute(1, 2, 0).cpu().numpy()  # Convert torch tensor to numpy array
        axes[i].imshow(image)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sys.path.append("src/taming-transformers/") # Add taming lib
    config_path = "configs/sampling_config_template.yaml"

    config = OmegaConf.load(config_path)
    opt = config.sampling_stuff

    ckpt_path = opt.ckpt_path
    config.model.params.ckpt_path = ckpt_path
    model = instantiate_from_config(config.model).to(device)
    model.eval()
    sampler = DDIMSampler(model)

    caption_generator = instantiate_from_config(opt.caption_config)
    # Readable averages:  [0.0037, 0.0001, 0.0271, 0.4418, 0.0121, 0.0219, 0.376, 0.1172]
    caption = caption_generator.generate()
    print("Caption used:", caption)

    def get_unconditional_token(batch_size):
        return [""] * batch_size

    def get_conditional_token(batch_size, caption):
        return [caption] * batch_size

    batch_size = opt.batch_size
    shape = [3, opt.size, opt.size] # shape of sampled imgs, CHW

    # scale of classifier free guidance
    scale = 1.5

    if opt.same_x_T:
        print("Using same starting noise (x_T).")
        x_T = torch.randn([batch_size] + shape, device=device)
    else:
        x_T = None

    with torch.no_grad():
        # unconditional token for classifier free guidance
        ut = get_unconditional_token(batch_size)
        uc = model.get_learned_conditioning(ut).to(torch.float32)

        ct = get_conditional_token(batch_size, caption)
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
            x_T = x_T,
            log_every_t=opt.log_every_t
        )

        if opt.show_intermediates:
            imgs = inters['pred_x0']

            x_imgs = []
            for img in imgs:

                x_img = model.decode_first_stage(img)
                x_img = torch.clamp((x_img + 1.0) / 2.0, min=0.0, max=1.0)
                x_img = (x_img * 255).to(torch.uint8).cpu()

                x_imgs.append(x_img)

            num_rows = len(x_imgs)
            num_cols = len(x_imgs[0])
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

            for i in range(num_rows):
                for j in range(num_cols):
                    axes[i, j].imshow(rearrange(x_imgs[i][j], pattern='c h w -> h w c'), cmap='gray')
                    axes[i, j].axis('off')

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()
        else:
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8).cpu()
            print(f"{x_samples_ddim.shape = }")

            grid = make_grid(x_samples_ddim, nrow=2)
            show_images(x_samples_ddim, title=caption)