#!/usr/bin/env bash

set -euo pipefail

source_image=$1
beaker_image=$2
beaker_workspace=$3
timestamp=$(date "+%Y%m%d%H%M%S")

beaker_user=$(beaker account whoami --format=json | jq -r '.[0].name')
beaker image create "${source_image}" --name "${beaker_image}-tmp" --workspace "${beaker_workspace}"
beaker image rename "${beaker_user}/${beaker_image}" "${beaker_image}-${timestamp}" 2>/dev/null || true
beaker image rename "${beaker_user}/${beaker_image}-tmp" "${beaker_image}"
