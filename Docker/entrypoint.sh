#!/bin/bash

# Variables
SECRET_NAME="deployment"
SECRET_KEY_GIT="git_pat"
SECRET_KEY_NEPTUNE="neptune_token"
TABLE_NAME="deployment"
TABLE_KEY="deployment_key"
echo "Deployment environment: $DEPLOYMENT_ENV"

# Set AWS region
export AWS_DEFAULT_REGION=eu-west-1
export AWS_REGION=eu-west-1

# Activate the conda aws environment.
source activate aws

# Get Github PAT
echo "Retrieving Github PAT from secret name/key: $SECRET_NAME / $SECRET_KEY_GIT ..."
GITHUB_PAT=$(aws secretsmanager get-secret-value --secret-id "$SECRET_NAME" | python -c "import sys, json; print(json.loads(json.load(sys.stdin)['SecretString'])['$SECRET_KEY_GIT'])")
echo "Done."

# Get repo config
echo "Retrieve repository config from environment/table/key: $DEPLOYMENT_ENV / $TABLE_NAME / $TABLE_KEY ..."
DYNDB_DATA=$(aws dynamodb get-item --consistent-read --table-name "$TABLE_NAME" --key '{ "'"$TABLE_KEY"'": {"S": "'"$DEPLOYMENT_ENV"'"} }')
python -c "import json; data = dict(${DYNDB_DATA}); data_filt = {k: v['S'] for k, v in data['Item']['repos']['M'].items()}; [print(k, v) for k, v in data_filt.items()]" > repo.txt
declare -A repo_dict
while IFS=' ' read -r key value; do
    repo_dict[$key]=$value
done < repo.txt
echo $repo_dict
echo "Done."

# Clone each repo
echo "Cloning repositories ..."
mkdir "$HOME/code"
for key in "${!repo_dict[@]}"; do
    echo "Cloning $key:${repo_dict[$key]} ..."
    git clone --quiet --recurse-submodules --branch "${repo_dict[$key]}" "https://$GITHUB_PAT@github.com/aiosyn/$key.git" "$HOME/code/$key"
done
echo "Cloning MeesMeuwissen/generationLDM ..."
git clone --quiet --recurse-submodules https://github.com/MeesMeuwissen/generationLDM "$HOME/code/generationLDM"
echo "Done."

# Get the Aiosyn repositories and add them to the python search path.
for name in $(ls ./code); do echo ${PWD}/code/${name} >> ${PWD}/miniconda3/envs/core/lib/python3.8/site-packages/aiosyn.pth; done

# Neptune token
echo "Retrieving Neptune token from secret name/key: $SECRET_NAME / $SECRET_KEY_NEPTUNE ..."
NEPTUNE_TOKEN=$(aws secretsmanager get-secret-value --secret-id "$SECRET_NAME" | python -c "import sys, json; print(json.loads(json.load(sys.stdin)['SecretString'])['$SECRET_KEY_NEPTUNE'])")
echo "Done."
export NEPTUNE_API_TOKEN=${NEPTUNE_TOKEN}

# Activate the conda environment.
source activate ${AIOSYN_CONDA_ENV}

# Make sure that the right shared objects are loaded.
source ./preload.sh

# Print some basic information for possible debug purposes.
echo "User: $(whoami)"
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo "Environment: ${CONDA_DEFAULT_ENV}"
echo "Activated environment: ${AIOSYN_CONDA_ENV}"
echo "Command: ${*}"

# exec conda run --name="${AIOSYN_CONDA_ENV}" --no-capture-output "${@}"

# With exec the new command takes control of this shell instead of creating a new one (useful for SIGINT).
exec "${@}"