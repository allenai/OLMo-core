#!/usr/bin/env bash

set -euo pipefail

name=$1
workspace=$2
full_name=$(beaker workspace images "${workspace}" --format=json | jq -r ".[] | select(.name==\"${name}\") | .fullName")
echo "${full_name}"
