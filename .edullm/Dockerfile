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
# The claim was that installing was impossible here: ai2-olmo-core declares torch>=2.6.0,
# so `pip install .` would fetch a multi-gigabyte wheel and take the build past what a
# standard runner can do, and `pip install --no-deps .` would leave an installed
# distribution whose imports fail anyway. Putting the source on PYTHONPATH was the answer.
#
# Both halves of that were measured in Phase 4 and the first half is false. A build of this
# file against the registered python base produced a working 2.94 GB image on an ordinary
# runner, and the torch wheel PyPI serves for linux x86_64 is CUDA-enabled, so it runs on a
# GPU. The install the comment was avoiding is affordable and always was.
#
# WHAT THE PLATFORM REGISTERS IS STILL A BARE PYTHON BASE, AND THAT IS A KNOWN LIMITATION
# RATHER THAN AN OVERSIGHT. The obvious alternative was to register a base that already
# carries torch, so the platform's pinned base digest would cover it. That was tried and
# withdrawn: the platform's Phase 1 rebuild study is asserted to have been made from the
# currently registered base, and re-recording it needs the new image published, which needs
# the registry change merged -- the change that invalidates the study is a prerequisite for
# the evidence that repairs it. The ways through were a knowingly broken main or relaxing a
# control and restoring it afterwards, and neither is worth a tidier narrative.
#
# So the honest statement is: the registered base digest describes a small fraction of this
# image, and the rest is pinned here instead. Reproducibility is preserved -- an exact
# version against an index that never republishes a version is as deterministic as a digest
# -- but a reader of the provenance record who takes base_image_digest as a description of
# the image will be wrong, and this paragraph is where they find out.
ENV PYTHONPATH=/opt/olmo-core/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The base sets no PYTHONPATH of its own. If a future base does set one, this line silently
# discards it. Prepending is not the safe-looking fix it appears to be:
# ENV PYTHONPATH=/opt/olmo-core/src:${PYTHONPATH} leaves a trailing colon when the base
# sets nothing, and Python reads a trailing colon as the current working directory.

WORKDIR /opt/olmo-core

# The build context is the repository root, so the root .dockerignore is what keeps .git
# out of this layer, and out of the context entirely.
COPY . .

# TORCH IS PINNED TO AN EXACT VERSION AND INSTALLED FIRST, WHICH IS THE WHOLE ARGUMENT.
#
# The project declares torch>=2.6.0. Resolved against PyPI at build time that is an
# unpinned multi-gigabyte dependency, so two builds of one commit months apart produce
# different images -- and the platform has a study whose entire subject is what does and
# does not stay identical across rebuilds. An exact version fixes that: a published wheel
# is never replaced, so `torch==2.6.0` names one artifact forever.
#
# 2.6.0 rather than the newest available, because it is the floor the project declares. A
# newer torch is a research decision with a compatibility surface, not something a
# Dockerfile should pick silently on everyone's behalf.
#
# Installed before the project so pip sees the constraint already satisfied and has no
# reason to resolve torch again.
#
# wandb comes from the extra the project already declares. boto3 is named directly because
# olmo_core.io imports it lazily for s3:// paths and the project declares it only under the
# dev extra, which would drag in pytest, mypy, sphinx and the rest of a contributor's
# toolchain -- none of which belongs in an image an untrusted training command runs inside.
#
# THE ASSERTION IS THE POINT OF THIS STEP, not the install. pip is free to decide some
# dependency's metadata is better served by a different torch and replace a CUDA build with
# a CPU wheel. Nothing would fail: the image builds, the container starts, the job runs, and
# it trains on the CPU at GPU prices while every log line looks ordinary. torch.version.cuda
# is None on a CPU build and a string on a CUDA one, so this turns the most expensive silent
# failure available here into a failed build.
#
# The version is re-asserted alongside it, because the pin above only binds if nothing later
# in the resolution moved it.
#
# The last line is the cheaper half and still worth having: it proves the three imports this
# image exists to make actually resolve, in the image, rather than at run time on a paid
# instance.
RUN set -eux; \
    python -m pip install --no-cache-dir "torch==2.6.0"; \
    python -m pip install --no-cache-dir ".[wandb]" boto3; \
    python -c "import torch; assert torch.__version__.startswith('2.6.0'), f'torch was moved off the pin: {torch.__version__}'"; \
    python -c "import torch; assert torch.version.cuda, 'the CUDA build of torch was replaced by a CPU wheel'"; \
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
