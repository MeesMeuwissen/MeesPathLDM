from aiosynawsmodules.services.sso import set_sso_profile
import logging

from aiosynawsmodules.services.batch import submit_batch_job

set_sso_profile(profile_name="aws-aiosyn-workloads-dev", region_name="eu-west-1")  #What is the correct profile?
logging.basicConfig(level=logging.INFO)

submit_batch_job(
    name="generation_sample",
    script_path= "generationLDM/sampling/generate_synthetic_dataset.py", #The script to be executed
    timeout_min=60 * 5, #Time in minutes
    gpu=True,
    account="computing-feature2", #For me, this is 2
    gpus=0,
    location='remote',
    #model="s3://aiosyn-data-eu-west-1-bucket-ops/models/generation/logs/03-11T16-54_unconditional_aiosyn_data_plip-GEN-301/last.ckpt",
    model= "s3://aiosyn-data-eu-west-1-bucket-ops/models/generation/unet/pathldm/epoch_3-001.ckpt",
    summary="-",
    number=1500, #How many imgs to generate
    all=False, #upload a zip of all images to S3. A subsample of 10 will always be uploaded
    n_attempts=3, #retries if it fails.
)