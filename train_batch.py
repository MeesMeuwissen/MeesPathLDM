import logging
from datetime import datetime

from aiosynawsmodules.services.batch import submit_batch_job
from aiosynawsmodules.services.sso import set_sso_profile

set_sso_profile(profile_name="aws-aiosyn-workloads-dev", region_name="eu-west-1")  #What is the correct profile?
logging.basicConfig(level=logging.INFO)

submit_batch_job(
    name="generation_train",
    script_path= "generationLDM/main.py", #The script to be executed
    timeout_min=60 * 48, #Time in minutes
    gpu=True,
    account="computing-feature2", #For me, this is 2
    gpus=0,
    location='remote',
    train=True,
    n_attempts=4,
    base='code/generationLDM/configs/latent-diffusion/text_cond/imagenet_finetune.yaml', #The correct config file
    neptune_mode='async'
)
print(f"Submitted at {datetime.now()}")