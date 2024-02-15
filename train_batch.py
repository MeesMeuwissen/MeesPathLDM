from aiosynawsmodules.services.sso import set_sso_profile
import logging

from aiosynawsmodules.services.batch import submit_batch_job

set_sso_profile(profile_name="aws-aiosyn-workloads-dev", region_name="eu-west-1")  #What is the correct profile?
logging.basicConfig(level=logging.INFO)

submit_batch_job(
    name="generation_train",
    script_path= "generationLDM/main.py", #The script to be executed
    timeout_min=60 * 20,
    gpu=True,
    account="computing-feature2", #Voor mij is dit 2
    gpus=0,
    base='code/generationLDM/configs/latent-diffusion/text_cond/unconditional_aiosyn_data.yaml', #The correct config file
    neptune_mode='async'
)