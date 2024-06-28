from datetime import datetime

from aiosynawsmodules.services.sso import set_sso_profile
import logging

from aiosynawsmodules.services.batch import submit_batch_job

set_sso_profile(profile_name="aws-aiosyn-workloads-dev", region_name="eu-west-1")  #What is the correct profile?
logging.basicConfig(level=logging.INFO)

submit_batch_job(
    name="generation_sample",
    script_path= "generationLDM/sampling/generate_synthetic_dataset.py", #The script to be executed
    timeout_min=60 * 20, #Time in minutes
    gpu=True,
    account="computing-feature2", #For me, this is 2
    gpus=0,
    location='remote',
    index=4, # Only used when uploading to the same dir with multiple samplers, i.e. 's3_directory' is set up in the config.
    config_path="code/generationLDM/sampling/configs/sampling_config_large.yaml",
    save_s3=True,
)

print(f"Submitted at {datetime.now()}")
text = " DO IT NOW! "
print(f"Did you forget to push? \n{text:=^23}")

'''
LET OP: Zorg dat de juiste mu, sig worden geladen. Dit gebeurt in ddpm.py, zoek op "FID_full". 
Dit is afhankelijk van de dataset gebruikt om te trainen. 
'''