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
# The claim was that installing was impossible here: ai2-olmo-core declares a torch
# dependency, so `pip install .` would fetch a multi-gigabyte wheel and take the build past what a
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

# A C COMPILER, BECAUSE torch.compile IS UNUSABLE WITHOUT ONE AND THE EXAMPLE TURNS IT ON.
#
# The registered base is a bare python image and carries no toolchain. That is invisible
# until something compiles, and the first thing that does is Inductor: a training run built
# from src/examples/llm/train.py sets compile_model=True, and the run dies with
# `InductorError: RuntimeError: Failed to find C compiler. Please specify via CC
# environment variable or set triton.knobs.build.impl.`
#
# Found the way these things are: a twelve-hour submission on 2026-08-01 reached the GPU,
# built the model, wrote its first checkpoint object to S3, and died on the first compiled
# region. Not at build time, not in a smoke test, and not until an A10G had been paid for.
#
# gcc rather than build-essential. Inductor and Triton shell out to a C compiler to build
# the kernels they generate; they do not need make, dpkg-dev, or the rest of the
# metapackage, and this image is what an untrusted training command runs inside -- so the
# smaller toolchain is both lighter and less to reason about. g++ comes with it because
# Inductor's CPU path emits C++.
#
# Pinned to no version deliberately, unlike torch above. The pin on torch exists because a
# resolved wheel changes what the model computes; a compiler that builds a kernel and then
# is never shipped anywhere does not, and pinning to a Debian point release would make this
# line expire on the base's next security update.
#
# apt-get is the right tool here rather than an assumption: the registered base digest
# sha256:57cd7c3a…10de resolves to Debian trixie carrying python 3.12.13, read out of the
# image config on Docker Hub rather than inferred from the repository name. An Alpine base
# would need apk and would not have manylinux wheels for torch either, so a base swap that
# broke this line would already have broken the install above it.
#
# `cc --version` at the end of the RUN is the assertion, in the same spirit as the torch
# checks below: it fails the build if the compiler is somehow not on PATH afterwards, which
# is the whole condition this layer exists to guarantee.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends gcc g++; \
    rm -rf /var/lib/apt/lists/*; \
    cc --version

# THE DEPENDENCIES ARE INSTALLED ABOVE THE SOURCE COPY, AND THE REASON IS MEASURED.
#
# This step used to sit below `COPY . .`, which put the four-gigabyte dependency install
# downstream of every source edit. Docker invalidates a layer and everything after it when
# an input changes, so a one-line change to a training script rebuilt the whole tree and
# pushed a four-gigabyte layer that deduplicated against nothing already in the registry.
# Read out of ECR on 2026-08-05: 59 images of this repository, 243.52 GiB stored, 2.39 GiB
# of it shared, and the per-image dependency layer distinct in all 59 -- 4335.7 MiB apiece,
# differing only because the commit differed.
#
# Above the copy the layer is keyed on pyproject.toml and on this file, so it is shared by
# every image built from one dependency set and rebuilt only when the dependencies
# themselves move. edullm-data is the control: its dependency install has been above its
# source copy all along, and its 40.0 MiB dependency layer is one blob shared by every
# image that runs it, where the same repository's earlier `pip install .` below the copy
# produced twelve distinct 36.2 MiB blobs for twelve builds.
#
# ONLY pyproject.toml IS COPIED HERE, WHICH IS THE POINT RATHER THAN AN ECONOMY. Copying
# more of the tree would key this layer on files that do not determine the dependency set,
# and each one would be another way for a source edit to throw the install away again.
#
# THE REQUIREMENTS ARE READ OUT OF pyproject.toml RATHER THAN RESTATED HERE. Writing the
# thirteen dependencies and the wandb extra into this file would duplicate the manifest and
# let the copy drift from it in silence -- a dependency added to the project would simply
# not be installed in this layer, and would be picked up later by the project install
# below, quietly restoring the four-gigabyte layer this step exists to prevent. tomllib is
# in the standard library at the 3.12.13 the registered base carries, so reading the
# manifest costs no package.
#
# The project itself is deliberately NOT installed here. Installing it would need the
# source, which is the thing this step is above, and a stub build would put an
# `ai2-olmo-core 2.5.0` on disk that the real install below would then find already
# satisfied and skip -- leaving an image whose package list looks right and whose code is
# absent.
COPY pyproject.toml ./

# TORCH IS PINNED TO AN EXACT VERSION AND INSTALLED FIRST, WHICH IS THE WHOLE ARGUMENT.
#
# The project declares a lower bound on torch. Resolved against PyPI at build time that is
# an unpinned multi-gigabyte dependency, so two builds of one commit months apart produce
# different images -- and the platform has a study whose entire subject is what does and
# does not stay identical across rebuilds. An exact version fixes that: a published wheel
# is never replaced, so an exact pin names one artifact forever.
#
# The pin moved twice before it was right, and both moves were measurements rather than
# preferences. It was 2.6.0, the floor this project declared, and the first GPU training
# run proved that floor does not import. `import olmo_core` reaches
# nn/attention/backend.py, which decorates a method with torch.compiler.disable(reason=...),
# and that keyword does not exist in 2.6.0 -- so the import dies with `TypeError: disable()
# got an unexpected keyword argument` before any model is built. The declared floor was
# stale metadata rather than a tested claim, and pyproject.toml is corrected with this.
#
# WHICH VERSION, MEASURED RATHER THAN GUESSED THE THIRD TIME. torch.compiler.disable grew
# its `reason` keyword in 2.8.0 exactly -- read out of torch/compiler/__init__.py at tags
# v2.7.1, v2.8.0, v2.9.0 and v2.10.0 -- so 2.8.0 is the true floor and everything below it
# fails the import outright.
#
# The pin is 2.9.0 rather than 2.8.0, and the difference is what is tested versus what is
# merely sufficient. .github/workflows/docker.yml builds this project against 2.7.1 and
# 2.9.0; of those, 2.9.0 is the only one that can import the package at all. So 2.9.0 is
# the lowest version that both works and is exercised by this repository, and 2.8.0 would
# be a floor nothing here has ever run.
#
# That the 2.7.1 leg of that matrix cannot import olmo_core.nn is a real inconsistency in
# this project and is worth someone looking at; it is not fixed here.
#
# This is also the argument for pinning rather than tracking a floor. An unpinned
# `>=2.6.0` resolves to whatever is newest, would have worked here by accident, and would
# have left the stale floor to be discovered by whoever next had an older environment.
#
# Installed before the project so pip sees the constraint already satisfied and has no
# reason to resolve torch again.
#
# wandb comes from the extra the project already declares. boto3 is named directly because
# olmo_core.io imports it lazily for s3:// paths and the project declares it only under the
# dev extra, which would drag in pytest, mypy, sphinx and the rest of a contributor's
# toolchain -- none of which belongs in an image an untrusted training command runs inside.
#
# botocore's `crt` EXTRA IS BESIDE boto3 BECAUSE THE CHECKSUM A CHECKPOINT WRITE ASKS FOR IS
# NOT IN botocore ITSELF. The platform commits a checkpoint by putting the payload with
# `ChecksumAlgorithm="CRC32C"` and then a `_SUCCESS` object carrying the digest S3 attested
# back; its reader compares the two and calls a checkpoint it cannot verify corrupt. So the
# algorithm is load-bearing rather than incidental, and CRC-32C is the one S3 offers as a
# whole-object checksum on a multipart upload where SHA-256 is composite-only. botocore has
# no pure-Python CRC-32C: `botocore.httpchecksum` adds it to the supported set only when
# `awscrt` imports, which is what the extra installs.
#
# Found the way these things are, again, and at the end rather than the start. Run
# run_019fcf53 on 2026-08-05 pulled the image, built the model, trained twenty steps on a
# T4, logged every one of them to W&B, and then died on the first line after the training
# loop with `botocore.exceptions.MissingDependencyException: Missing Dependency: Using
# CRC32C requires an additional dependency. You will need to pip install botocore[crt]
# before proceeding.` A whole run's compute, and nothing kept.
#
# NAMED HERE RATHER THAN LEFT TO boto3, which depends on botocore without extras, and no
# other dependency in this image asks for this one. It sits on the same pip invocation as
# boto3 deliberately: one resolution then picks the botocore both agree on, where a separate
# later install can move botocore under boto3's feet.
#
# UNPINNED, LIKE boto3 ABOVE AND UNLIKE torch BELOW, AND THE LINE BETWEEN THEM IS WHAT A
# RESOLVED VERSION CHANGES. A different torch changes what the model computes and so what a
# rebuild of one commit produces. A different awscrt changes how fast a CRC-32C is arrived
# at, not what it is -- the value is Castagnoli's and is fixed by the standard, which is why
# the check value below can be written down here at all.
#
# THE DATASET READER IS INSTALLED HERE BECAUSE A TRAINING RUN CANNOT OPEN A CORPUS WITHOUT
# IT. `edullm-data` publishes the platform corpora into s3://edullm-data behind a validator,
# and `edullm_data.read` is the supported way to turn a dataset id and version into file URIs
# plus the dtype those files actually hold. Without it in the image, a container told which
# corpus it may read has no way to read it.
#
# PINNED BY COMMIT, AND TO THE SAME COMMIT THE PLATFORM PINS. The platform declares this
# reader in its own `reader` dependency group at 38bf831a, and the reasoning for pinning a
# commit rather than a tag lives there: `is_trainable` does not exist at v0.2.0, so a tag pin
# answers a request for training data with held-out shards. Two pins on one commit is a
# duplication worth having -- the platform resolves it for its tests and this image resolves
# it for the run -- but they must move together, and a mismatch is a container reading
# through a different library than the one the tests exercised.
#
# A TARBALL URL RATHER THAN git+https, AND NOT FOR STYLE. `pip install git+https://…` shells
# out to git, and the registered base is a bare python image that carries none. The choice
# was between adding git to the image and asking pip for the archive GitHub already serves
# for any commit; the archive needs no new package in an image an untrusted training command
# runs inside. The URL carries the same forty hex digits, so it names exactly one tree just
# as immutably as the git form does.
#
# The import assertion below is the same argument as the one above it: `import edullm_data`
# would prove the distribution installed, and `from edullm_data.read import dataset_paths`
# proves the module a training run actually reaches for loads.
RUN set -eux; \
    python -m pip install --no-cache-dir "torch==2.9.0"; \
    python -c "import tomllib; project = tomllib.load(open('pyproject.toml', 'rb'))['project']; open('/tmp/dependencies.txt', 'w').write('\n'.join(project['dependencies'] + project['optional-dependencies']['wandb']) + '\n')"; \
    python -m pip install --no-cache-dir -r /tmp/dependencies.txt boto3 "botocore[crt]"; \
    rm /tmp/dependencies.txt; \
    python -m pip install --no-cache-dir "edullm-data @ https://github.com/edu-llm/edullm-data/archive/38bf831a6c3f445e394784018441fd59288b876c.tar.gz"

# The build context is the repository root, so the root .dockerignore is what keeps .git
# out of this layer, and out of the context entirely.
COPY . .

# THE PROJECT ITSELF, AFTER ITS SOURCE AND AFTER ITS DEPENDENCIES, AND THE RESOLUTION IS
# THE SAME ONE IT ALWAYS WAS. This is still `.[wandb]` rather than `--no-deps .`, which
# would be the faster spelling and the wrong one: --no-deps would install the project
# without checking that the layer above actually covered its requirements, so a dependency
# added to pyproject.toml and missed by the extraction would go absent from the image
# instead of being installed here. Asking for the extra means pip verifies the whole set,
# finds it satisfied by the layer above, and reaches the network for nothing -- and if it
# ever does reach the network, that is the extraction having drifted and worth the cost.
#
# The layer this produces is the project's own files. Measured on 2026-08-05 by building
# this file twice against the registered base with only a source file changed between the
# two, and pushing both to a registry: the second image added 4.4 MiB of blobs, where the
# combined step below the source copy wrote 4335.7 MiB that shared with nothing. The second
# build also took 26 seconds against roughly 400 for the first, because the dependency
# layer above is a cache hit rather than a re-resolution.
#
# THE ASSERTIONS RUN HERE, BELOW THE LAST INSTALL, WHICH IS THE ONLY PLACE THEY MEAN
# ANYTHING. Every one of them describes the state the image ships in, and this install is
# the last step that can change it: pip resolving the project is free to move a dependency
# the layer above put there. An assertion above the copy would be checking a state no
# container ever runs.
#
# THE CUDA CHECK IS THE POINT OF THE PAIR, not the install. pip is free to decide some
# dependency's metadata is better served by a different torch and replace a CUDA build with
# a CPU wheel. Nothing would fail: the image builds, the container starts, the job runs, and
# it trains on the CPU at GPU prices while every log line looks ordinary. torch.version.cuda
# is None on a CPU build and a string on a CUDA one, so this turns the most expensive silent
# failure available here into a failed build.
#
# The version is re-asserted alongside it, because the pin above only binds if nothing later
# in the resolution moved it.
#
# THE SUBMODULE IMPORT IS THE LAST LINE'S SHALLOW VERSION MADE DEEP, AND IT ALREADY FAILED ONCE.
#
# It used to read `import olmo_core, boto3, wandb`, which passed on an image whose first
# GPU training run then died at startup: the top-level package imports lazily, so nothing
# under olmo_core.nn was touched until the workload asked for a model, four minutes and one
# image pull later, on a paid instance. A check that stops at the package name proves the
# distribution installed, not that the code loads.
#
# So it names what a training workload actually reaches for. Anything that breaks between
# the pinned torch and this library now breaks the build, where it costs a runner minute.
#
# THE CRC-32C LINE IS THE FAILED RUN, MADE INTO A BUILD STEP, AND IT IS TWO CHECKS BECAUSE
# THERE ARE TWO WAYS TO GET THIS WRONG. `pip install botocore[crt]` succeeding proves a
# distribution is on disk; it does not prove botocore reached it. botocore imports awscrt
# inside a try/except and sets HAS_CRT False on any failure -- an incompatible version, a
# wheel built for another libc -- so an image can carry awscrt and refuse CRC32C anyway,
# silently, until a training run has already been paid for.
#
# The first check is therefore the exact call that died: put_object with
# ChecksumAlgorithm="CRC32C". Stubber answers it, so it costs no network, no credentials and
# no bucket -- but botocore resolves the checksum before the stub is consulted, which is
# precisely the frame the run's traceback names, so the check raises the run's own
# MissingDependencyException when the CRT is not reachable. Verified both ways on
# 2026-08-05: it raises on the image published from this file's parent commit and passes
# once the extra is installed.
#
# The second is awscrt's own answer to CRC-32C of b"123456789", which the standard fixes at
# 0xE3069283 and the platform's checkpoints.py asserts against the same constant. It costs a
# microsecond and it is the difference between "botocore will use this" and "and the number
# it gets is right".
RUN set -eux; \
    python -m pip install --no-cache-dir ".[wandb]"; \
    python -c "import torch; assert torch.__version__.startswith('2.9.0'), f'torch was moved off the pin: {torch.__version__}'"; \
    python -c "import torch; assert torch.version.cuda, 'the CUDA build of torch was replaced by a CPU wheel'"; \
    python -c "import boto3, wandb; from olmo_core.data import TokenizerConfig; from olmo_core.nn.transformer import TransformerConfig; print(TransformerConfig.olmo2_190M(vocab_size=TokenizerConfig.gpt2().padded_vocab_size()).__class__.__name__)"; \
    python -c "import boto3; from botocore import UNSIGNED; from botocore.config import Config; from botocore.stub import Stubber; client = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED)); stub = Stubber(client); stub.add_response('put_object', {}); stub.activate(); client.put_object(Bucket='b', Key='k', Body=b'123456789', ChecksumAlgorithm='CRC32C')"; \
    python -c "from awscrt.checksums import crc32c; assert crc32c(b'123456789') == 0xE3069283, 'awscrt computes something other than CRC-32C'"; \
    python -c "from edullm_data.read import dataset_paths, resolve_latest; print(dataset_paths.__module__)"

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
