#!/usr/bin/env bash

set -exuo pipefail

beaker_workspace=$1

beaker_user=$(beaker account whoami --format=json | jq -r '.[0].name')

beaker secret write -w "${beaker_workspace}" "${beaker_user}_GITHUB_TOKEN" $GITHUB_TOKEN
beaker secret write -w "${beaker_workspace}" "${beaker_user}_HF_TOKEN" $HF_TOKEN
beaker secret write -w "${beaker_workspace}" "${beaker_user}_AWS_CONFIG" $AWS_CONFIG
beaker secret write -w "${beaker_workspace}" "${beaker_user}_AWS_CREDENTIALS" $AWS_CREDENTIALS
# beaker secret write -w "${beaker_workspace}" "${beaker_user}_AWS_ACCESS_KEY_ID" $AWS_ACCESS_KEY_ID
# beaker secret write -w "${beaker_workspace}" "${beaker_user}_AWS_SECRET_ACCESS_KEY" $AWS_SECRET_ACCESS_KEY
beaker secret write -w "${beaker_workspace}" "${beaker_user}_BEAKER_TOKEN" $BEAKER_TOKEN
beaker secret write -w "${beaker_workspace}" "${beaker_user}_WANDB_API_KEY" $WANDB_API_KEY
beaker secret write -w "${beaker_workspace}" "${beaker_user}_R2_ACCESS_KEY_ID" $R2_ACCESS_KEY_ID
beaker secret write -w "${beaker_workspace}" "${beaker_user}_R2_SECRET_ACCESS_KEY" $R2_SECRET_ACCESS_KEY
beaker secret write -w "${beaker_workspace}" "${beaker_user}_R2_ENDPOINT_URL" $R2_ENDPOINT_URL