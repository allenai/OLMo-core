#!/usr/bin/env bash
#
# This script sets up the environment needed inside a container and then runs the command passed in.
#

# Put setup of conda in an env variable if conda is needed
if [[ ! -z "${CONDA_ENV}" ]]; then
  source /opt/miniconda3/bin/activate ${CONDA_ENV}
fi

${@}
