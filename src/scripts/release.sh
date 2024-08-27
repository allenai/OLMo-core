#!/bin/bash

set -e

TAG=$(python -c 'from olmo_core.version import VERSION; print("v" + VERSION)')

git pull
python src/scripts/prepare_changelog.py

read -rp "Creating new release for $TAG. Do you want to continue? [Y/n] " prompt

if [[ $prompt == "y" || $prompt == "Y" || $prompt == "yes" || $prompt == "Yes" ]]; then
    git add -A
    git commit -m "(chore) prepare for release $TAG" || true && git push
    echo "Creating new git tag $TAG"
    git tag "$TAG" -m "$TAG"
    git push --tags
else
    echo "Canceled"
    git checkout CHANGELOG.md
    exit 1
fi
