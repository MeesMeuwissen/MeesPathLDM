import logging
from datetime import datetime

from aiosynawsmodules.services.batch import submit_batch_job
from aiosynawsmodules.services.sso import set_sso_profile

set_sso_profile(profile_name="aws-aiosyn-workloads-dev", region_name="eu-west-1")  #What is the correct profile?
logging.basicConfig(level=logging.INFO)

submit_batch_job(
    name="generation_resume",
    script_path= "generationLDM/main.py", #The script to be executed
    timeout_min=60 * 48, #Time in minutes
    gpu=True,
    account="computing-feature2", #For me, this is 2
    gpus=0,
    location='remote',
    train=True,
    resume= "s3://aiosyn-data-eu-west-1-bucket-ops/models/generation/logs/03-18-remote-GEN-340/", #S3 url of logdir
    n_attempts=1,
    base='code/generationLDM/configs/latent-diffusion/text_cond/uncond_srikar_finetune.yaml', #Make sure this is correct!
    neptune_mode='async'
)
print(f"Submitted at {datetime.now()}")