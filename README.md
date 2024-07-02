Repository for Master's Thesis, conducted January - July 2024, in collaboration with Aiosyn.

Code heavily based on  [PathLDM: Text conditioned Latent Diffusion Model for Histopathology.](https://github.com/cvlab-stonybrook/PathLDM) 

## Environment
To run training, set up a conda environment by running 

`conda env create -f Docker/env_macos.yaml` 
if on macos or 

`conda env create -f Docker/env_linux.yaml` if on linux. 

Activate it with `conda activate generation`

## Checkpoints
For pretrained models, I refer to  [the original repo](https://github.com/cvlab-stonybrook/PathLDM)
Our model trained on kidney data can be found [here](https://drive.google.com/file/d/1OUgzuM9U8VKevXNQ8fjaOrTdp4JQxO2X/view?usp=drive_link).  

## Training
To start a training run, please provide your own dataloader, set it up in `configs/latent-diffusion/text_cond/local/config_template.yaml` and run the command

`python main_clean.py -t --gpus 0 --base configs/latent-diffusion/text_cond/local/config_template.yaml`

`main.py` was used by me and contains many more features, but is not runnable without access to Aiosyns code. `main_clean.py` should be easier to get running. 


## Sampling
To sample, set up `sampling/configs/sampling_config.yaml` with a ckpt and (optionally) a caption generator like found in `sampling/captions.py`,
and run 

`sample_conditional.py`


