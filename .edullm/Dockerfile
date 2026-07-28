# Research image for the eduLLM platform build pipeline.
#
# The platform supplies the registered base image as --build-arg BASE_IMAGE, already
# pinned to a digest. Its verify_dockerfile_base gate runs before the build and refuses
# this file unless BASE_IMAGE is declared in the global scope with no default and every
# stage roots at ${BASE_IMAGE}. The ARG is bare on purpose: a default would be a base
# image nobody registered, sitting here waiting for the day the build-arg is dropped,
# and the provenance record would still name the registered digest.
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

# THIS FILE USED TO INSTALL NOTHING, AND THE REASON IT STOPPED IS WORTH READING.
#
# Against the previous base -- a plain python image -- installing was not an option:
# ai2-olmo-core declares torch>=2.6.0, so `pip install .` would have fetched a
# multi-gigabyte wheel and taken the build past what a standard runner can do, and
# `pip install --no-deps .` would have left an installed distribution whose imports fail
# anyway, having bought nothing. Putting the source on PYTHONPATH was the honest answer.
#
# The comment that replaced said the code would "become importable with no edit here" on a
# training base that already carries torch. That was optimistic and Phase 4 measured it.
# The registered base is now pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime, which carries
# torch, numpy, pyyaml, filelock, requests and packaging -- and none of cached-path,
# dataclass-extensions, rich, safetensors, importlib_resources, bettermap or pandas, every
# one of which ai2-olmo-core imports. `import olmo_core` fails on that base without this
# step.
#
# So the install is back, and it is cheap in a way it could not have been before: torch is
# already present, so what arrives is roughly a couple of hundred megabytes of small pure
# Python packages rather than the wheel the old comment was avoiding.
ENV PYTHONPATH=/opt/olmo-core/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The base sets no PYTHONPATH of its own -- checked against the registered digest's image
# config rather than assumed, because the landmine recorded here previously was that this
# ENV overwrites rather than prepends. If a future base does set one, this line silently
# discards it. Prepending is not the safe-looking fix it appears to be:
# ENV PYTHONPATH=/opt/olmo-core/src:${PYTHONPATH} leaves a trailing colon when the base
# sets nothing, and Python reads a trailing colon as the current working directory.

WORKDIR /opt/olmo-core

# The build context is the repository root, so the root .dockerignore is what keeps .git
# out of this layer, and out of the context entirely.
COPY . .

# wandb comes from the extra the project already declares. boto3 is named directly because
# olmo_core.io imports it lazily for s3:// paths and the project declares it only under the
# dev extra, which would drag in pytest, mypy, sphinx and the rest of a contributor's
# toolchain -- none of which belongs in an image an untrusted training command runs inside.
#
# THE ASSERTION IS THE POINT OF THIS STEP, not the install. pip is free to decide that some
# dependency's metadata is better served by a different torch and replace the base's CUDA
# build with a CPU wheel from PyPI. Nothing would fail: the image builds, the container
# starts, the job runs, and it trains on the CPU at GPU prices while every log line looks
# ordinary. torch.version.cuda is None on a CPU build and a string on a CUDA one, so this
# turns the most expensive silent failure available here into a failed build.
#
# The second line is the cheaper half and still worth having: it proves the three imports
# this image exists to make actually resolve, in the image, rather than at run time on a
# paid instance.
RUN set -eux; \
    python -m pip install --no-cache-dir ".[wandb]" boto3; \
    python -c "import torch; assert torch.version.cuda, 'pip replaced the CUDA build of torch with a CPU wheel'"; \
    python -c "import olmo_core, boto3, wandb"

# No CMD. The entrypoint of the registered base is inherited deliberately, so swapping the
# base does not leave a stale command behind.
#
# No USER either, so this image runs as root, inherited from the base. That was recorded
# here as a decision for the phase that introduces the workload role and can settle it
# against a real runtime. Phase 4 is that phase and the answer is: still root, for now. The
# container's blast radius is bounded by the workload role rather than by its uid -- that
# role may write one team's output prefix and read it back, and holds nothing else -- and
# the training base expects to own /opt/conda, which a non-root uid cannot write to without
# a chown layer that would roughly double the image. Revisit when the image is built for a
# fixed uid rather than adapted to one.
