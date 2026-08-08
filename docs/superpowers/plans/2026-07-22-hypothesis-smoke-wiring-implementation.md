# Skill-DAG and Curriculum Smoke Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make commit `30322fd0c62deef799a3aca8fcef410186025a03` genuinely train on the
prepared Skill-DAG mixtures and curriculum orders, route runs to stable W&B studies, and preserve
the exact data condition used.

**Architecture:** Keep core OLMo unchanged. Add one shared data/provenance module, one curriculum
materializer, and one OLMo2-190M hypothesis smoke entrypoint. Existing shell/PowerShell helpers
prepare the condition, perform a dry-run, then execute training. Unit tests prove the generated
mix/order reaches the resolved loader configuration before any GPU run.

**Tech Stack:** Python, NumPy memmaps, OLMo composable data APIs, OLMo2-190M, W&B, pytest, Bash,
PowerShell, Slurm/Engaging.

## Global Constraints

- Start from the verified Plan 2 tip, then cherry-pick
  `2129f69664ff355ea209970bfeb842bf5436f8fb` and
  `30322fd0c62deef799a3aca8fcef410186025a03`. Resolve conflicts without dropping queue/Agent
  Skill files, rerun all Plan 1/2 tests, and obtain review for the resulting new SHA.
- Do not modify `src/olmo_core/**`.
- Do not modify the generic `OLMo2-190M.py`/`OLMo2-370M.py` templates.
- Use OLMo2-190M for hypothesis smokes.
- Dry-run must succeed before training, but success must not suppress training.
- No AI2 GCS path may appear in a resolved ORCD smoke config.
- W&B entity is `eduLLM`, project is `pretraining`, groups are stable studies.
- A smoke verifies data wiring and execution, not the hypothesis effect.

---

## File Structure

Create:

```text
src/scripts/hypothesis/shared/smoke_data.py
src/scripts/hypothesis/curriculum/materialize_ordered_shard.py
src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py
src/Dockerfile.orcd
src/scripts/orcd/build_apptainer.sh

src/test/scripts/hypothesis/
  conftest.py
  smoke_data_test.py
  materialize_ordered_shard_test.py
  resolved_config_test.py
  wrapper_contract_test.py
```

Modify:

```text
src/scripts/hypothesis/shared/prepare_smoke_dolma.py
src/scripts/hypothesis/skill_dag/run_smoke.sh
src/scripts/hypothesis/skill_dag/run_smoke.ps1
src/scripts/hypothesis/curriculum/run_smoke.sh
src/scripts/hypothesis/curriculum/run_smoke.ps1
src/scripts/hypothesis/README.md
src/scripts/hypothesis/CHANNEL_SMOKE.md
```

Generated output:

```text
runs/skilldag-smoke-natural/
  config.json
  metrics.json
  hypothesis_identity.json
  provenance/mix.json
  # or provenance/order.jsonl
```

---

### Task 1: Normalize Mix Weights and Build Skill-DAG Data

**Files:**
- Create: `src/scripts/hypothesis/shared/smoke_data.py`
- Create: `src/test/scripts/hypothesis/conftest.py`
- Test: `src/test/scripts/hypothesis/smoke_data_test.py`

**Interfaces:**
- `load_mix_weights(path) -> dict[str, float]`
- `build_skill_dag_data(common, data_dir, mix_path, seed) -> DataComponents`
- `condition_identity(...) -> dict`

- [ ] **Step 1: Write failing mix tests**

```python
# src/test/scripts/hypothesis/conftest.py
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def load_script_module():
    def load(name: str, path: str):
        spec = importlib.util.spec_from_file_location(name, Path(path))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
```

```python
# src/test/scripts/hypothesis/smoke_data_test.py
import json

import pytest

def test_load_mix_weights_accepts_both_schemas(tmp_path, load_script_module):
    module = load_script_module(
        "smoke_data", "src/scripts/hypothesis/shared/smoke_data.py"
    )
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"weights": {"web": 2, "code": 1}}))
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps({"initial_weights": {"web": 2, "code": 1}}))
    assert module.load_mix_weights(weights) == pytest.approx({"web": 2 / 3, "code": 1 / 3})
    assert module.load_mix_weights(initial) == pytest.approx({"web": 2 / 3, "code": 1 / 3})


def test_rejects_nonpositive_mix(tmp_path, load_script_module):
    module = load_script_module(
        "smoke_data", "src/scripts/hypothesis/shared/smoke_data.py"
    )
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"weights": {"web": 0, "code": 1}}))
    with pytest.raises(ValueError, match="positive"):
        module.load_mix_weights(path)
```

Add a helper in `conftest.py` that uses `importlib.util.spec_from_file_location()` because
`src/scripts` is not packaged.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -v src/test/scripts/hypothesis/smoke_data_test.py
```

Expected: FAIL because `smoke_data.py` does not exist.

- [ ] **Step 3: Implement normalized weight loading**

```python
# src/scripts/hypothesis/shared/smoke_data.py
import hashlib
import json
import math
from pathlib import Path

from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    MixingInstanceSourceConfig,
    MixingInstanceSourceSpecConfig,
)
from olmo_core.internal.experiment import CommonComponents, DataComponents


def load_mix_weights(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if all(isinstance(value, (int, float)) for value in data.values()):
        raw = data
    else:
        raw = data.get("weights", data.get("initial_weights"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("mix must contain non-empty 'weights' or 'initial_weights'")
    weights = {str(key): float(value) for key, value in raw.items()}
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("all mix weights must be finite and positive")
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
```

- [ ] **Step 4: Implement data-component construction**

```python
def build_skill_dag_data(
    common: CommonComponents,
    *,
    data_dir: Path,
    mix_path: Path,
    seed: int,
) -> DataComponents:
    weights = load_mix_weights(mix_path)
    index_path = data_dir / "manifests" / "domain_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing = sorted(set(weights) - set(index["domains"]))
    if missing:
        raise ValueError(f"mix domains missing from data: {missing}")

    specs = []
    for domain, ratio in weights.items():
        shards = index["domains"][domain]["shards"]
        if not shards:
            raise ValueError(f"domain '{domain}' has no tokenized shards")
        source = ConcatAndChunkInstanceSourceConfig.from_npy(
            *(str(data_dir / shard) for shard in shards),
            tokenizer=common.tokenizer,
            sequence_length=common.max_sequence_length,
            label=domain,
        )
        specs.append(
            MixingInstanceSourceSpecConfig(
                source=source,
                ratio=ratio,
                max_repetition_factor=4.0,
                label=domain,
            )
        )

    mixed = MixingInstanceSourceConfig(
        source_specs=specs,
        seed=seed,
        label=mix_path.stem,
    )
    loader = ComposableDataLoaderConfig(
        tokenizer=common.tokenizer,
        global_batch_size=common.global_batch_size,
        seed=seed,
        work_dir=common.work_dir,
        num_workers=2,
    )
    return DataComponents(dataset=[mixed], data_loader=loader)
```

- [ ] **Step 5: Add condition identity**

```python
def condition_identity(*, mode: str, data_dir: Path, artifact: Path, seed: int) -> dict:
    index = data_dir / "manifests" / "domain_index.json"
    return {
        "mode": mode,
        "substrate_version": json.loads(index.read_text())["substrate_version"],
        "domain_index_sha256": sha256_file(index),
        "artifact_path": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "seed": seed,
    }
```

Also write one queue-consumable manifest per condition:

```python
def file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


manifest = {
    "kind": "skill-dag",
    "data_dir": str(data_dir.resolve()),
    "mix_file": str(mix_path.resolve()),
    "files": [
        file_record(mix_path),
        *[file_record(path) for path in sorted(data_dir.glob("tokenized/*/*.npy"))],
    ],
}
```

Curriculum manifests use `kind: curriculum`, `data_dir`, `order_file`, and records for the order
file, source shards, and materialized shard. The queue request references this manifest and its own
measured SHA-256.

- [ ] **Step 6: Add data-component tests**

Build a two-domain temporary tree with raw uint32 shards and assert:

- The dataset is a one-element list containing `MixingInstanceSourceConfig`.
- Labels and ratios match the selected JSON.
- Loader paths are local.
- Ratios sum to one.

- [ ] **Step 7: Run tests**

```bash
pytest -v src/test/scripts/hypothesis/smoke_data_test.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/scripts/hypothesis/shared/smoke_data.py src/test/scripts/hypothesis
git commit -m "feat: connect Skill-DAG mixes to data loader"
```

---

### Task 2: Materialize Curriculum Order

**Files:**
- Create: `src/scripts/hypothesis/curriculum/materialize_ordered_shard.py`
- Modify: `src/scripts/hypothesis/shared/smoke_data.py`
- Test: `src/test/scripts/hypothesis/materialize_ordered_shard_test.py`

**Interfaces:**
- `materialize(data_dir, order_path, output_path) -> Path`
- `build_curriculum_data(common, ordered_shard, seed) -> DataComponents`

- [ ] **Step 1: Write failing materialization test**

```python
# src/test/scripts/hypothesis/materialize_ordered_shard_test.py
import json

import numpy as np

def test_materializes_requested_document_order(tmp_path, load_script_module):
    module = load_script_module(
        "materialize",
        "src/scripts/hypothesis/curriculum/materialize_ordered_shard.py",
    )
    shard = tmp_path / "tokenized" / "web" / "shard-00000.npy"
    shard.parent.mkdir(parents=True)
    array = np.memmap(shard, mode="w+", dtype=np.uint32, shape=(6,))
    array[:] = [10, 11, 20, 21, 22, 30]
    array.flush()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    rows = [
        {"doc_id": "a", "token_path": "tokenized/web/shard-00000.npy", "token_offset": 0, "token_length": 2},
        {"doc_id": "b", "token_path": "tokenized/web/shard-00000.npy", "token_offset": 2, "token_length": 3},
        {"doc_id": "c", "token_path": "tokenized/web/shard-00000.npy", "token_offset": 5, "token_length": 1},
    ]
    (manifests / "docs.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    order = tmp_path / "order.jsonl"
    order.write_text('{"doc_id":"c"}\n{"doc_id":"a"}\n{"doc_id":"b"}\n')
    output = tmp_path / "ordered.npy"
    module.materialize(tmp_path, order, output)
    assert np.memmap(output, dtype=np.uint32).tolist() == [30, 10, 11, 20, 21, 22]
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -v src/test/scripts/hypothesis/materialize_ordered_shard_test.py
```

- [ ] **Step 3: Implement exact document slicing**

```python
# src/scripts/hypothesis/curriculum/materialize_ordered_shard.py
import argparse
import json
from pathlib import Path

import numpy as np


def materialize(data_dir: Path, order_path: Path, output_path: Path) -> Path:
    docs = {}
    for line in (data_dir / "manifests" / "docs.jsonl").read_text().splitlines():
        row = json.loads(line)
        docs[row["doc_id"]] = row
    order = [json.loads(line)["doc_id"] for line in order_path.read_text().splitlines()]
    if set(order) != set(docs) or len(order) != len(docs):
        raise ValueError("order must contain every document exactly once")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = sum(int(docs[doc_id]["token_length"]) for doc_id in order)
    output = np.memmap(output_path, mode="w+", dtype=np.uint32, shape=(count,))
    cursor = 0
    for doc_id in order:
        row = docs[doc_id]
        source = np.memmap(data_dir / row["token_path"], dtype=np.uint32, mode="r")
        start = int(row["token_offset"])
        end = start + int(row["token_length"])
        output[cursor : cursor + end - start] = source[start:end]
        cursor += end - start
    output.flush()
    return output_path
```

- [ ] **Step 4: Build non-shuffled curriculum data**

Add to `smoke_data.py`:

```python
def build_curriculum_data(
    common: CommonComponents, *, ordered_shard: Path, seed: int
) -> DataComponents:
    source = ConcatAndChunkInstanceSourceConfig.from_npy(
        str(ordered_shard),
        tokenizer=common.tokenizer,
        sequence_length=common.max_sequence_length,
        label=ordered_shard.stem,
    )
    loader = ComposableDataLoaderConfig(
        tokenizer=common.tokenizer,
        global_batch_size=common.global_batch_size,
        seed=seed,
        work_dir=common.work_dir,
        shuffle=False,
        num_workers=2,
    )
    return DataComponents(dataset=[source], data_loader=loader)
```

- [ ] **Step 5: Run tests**

```bash
pytest -v src/test/scripts/hypothesis/materialize_ordered_shard_test.py \
  src/test/scripts/hypothesis/smoke_data_test.py
```

- [ ] **Step 6: Commit**

```bash
git add src/scripts/hypothesis src/test/scripts/hypothesis
git commit -m "feat: materialize curriculum smoke order"
```

---

### Task 3: Local Hypothesis Smoke Training Entrypoint

**Files:**
- Create: `src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py`
- Test: `src/test/scripts/hypothesis/resolved_config_test.py`

**Interfaces:**
- Consumes: `SMOKE_MODE`, `SMOKE_DATA_DIR`, `SMOKE_MIX_FILE` or `SMOKE_ORDER_FILE`,
  W&B routing variables.
- Produces: local-only config, 20-step train, W&B metrics, condition identity.

- [ ] **Step 1: Write failing resolved-config test**

Invoke the script with `dry_run` and a temporary Skill-DAG tree, then assert output/config:

- Contains only local paths.
- Contains `olmo2_190M`.
- Uses `ComposableDataLoaderConfig`.
- Uses `Duration.steps(20)`.
- Has no downstream or GCS evaluation callback.
- W&B entity/project/group match environment.

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -v src/test/scripts/hypothesis/resolved_config_test.py
```

- [ ] **Step 3: Implement a focused config builder**

Use:

```python
SEQ_LEN = int(os.environ.get("SMOKE_SEQUENCE_LENGTH", "512"))
STEPS = int(os.environ.get("SMOKE_STEPS", "20"))
GLOBAL_BATCH_SIZE = SEQ_LEN * 8
SEED = int(os.environ.get("SMOKE_SEED", "0"))
```

Build:

- `TransformerConfig.olmo2_190M(...)`.
- Torch attention backend unless explicitly overridden.
- `rank_microbatch_size=SEQ_LEN`.
- BF16 and one-device-compatible training.
- Short warmup: `CosWithWarmup(warmup_steps=max(1, min(5, STEPS // 4)))`.
- `TrainerConfig(max_duration=Duration.steps(STEPS), no_evals=True)`.
- `ConfigSaverCallback`.
- `MetricSaverCallback(fixed_steps=[STEPS])`.
- `WandBCallback` using `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_GROUP`.

Select data:

```python
if mode == "skill_dag":
    data_builder = partial(
        build_skill_dag_data,
        data_dir=data_dir,
        mix_path=Path(os.environ["SMOKE_MIX_FILE"]),
        seed=SEED,
    )
elif mode == "curriculum":
    data_builder = partial(
        build_curriculum_data,
        ordered_shard=Path(os.environ["SMOKE_ORDER_FILE"]),
        seed=SEED,
    )
else:
    raise ValueError(f"unknown SMOKE_MODE: {mode}")
```

Use `olmo_core.internal.experiment.main` with the custom `data_config_builder`; do not call
`build_default_data_components`.

- [ ] **Step 4: Save condition identity**

Add a tiny callback in the script (not core) that writes `hypothesis_identity.json` and copies the
mix/order artifact to `save_folder/provenance/` on rank zero before training. Include SHA-256,
substrate version, seed, mode, and source fingerprints.

- [ ] **Step 5: Run dry-run test**

```bash
pytest -v src/test/scripts/hypothesis/resolved_config_test.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py \
  src/test/scripts/hypothesis/resolved_config_test.py
git commit -m "feat: add 190M hypothesis smoke trainer"
```

---

### Task 4: Repair Shell and PowerShell Helpers

**Files:**
- Modify: `src/scripts/hypothesis/skill_dag/run_smoke.sh`
- Modify: `src/scripts/hypothesis/skill_dag/run_smoke.ps1`
- Modify: `src/scripts/hypothesis/curriculum/run_smoke.sh`
- Modify: `src/scripts/hypothesis/curriculum/run_smoke.ps1`
- Test: `src/test/scripts/hypothesis/wrapper_contract_test.py`

**Interfaces:**
- Consumes: data directory, condition, private W&B environment.
- Produces: successful dry-run followed by real training.

- [ ] **Step 1: Write regression tests**

```python
# src/test/scripts/hypothesis/wrapper_contract_test.py
from pathlib import Path


def test_skill_dag_runs_dry_run_then_train():
    text = Path("src/scripts/hypothesis/skill_dag/run_smoke.sh").read_text()
    assert " dry_run " in text
    assert " train_single " in text
    assert "||" not in text
    assert "SMOKE_MIX_FILE" in text
    assert "SMOKE_RUN_MODE" in text


def test_curriculum_materializes_and_trains():
    text = Path("src/scripts/hypothesis/curriculum/run_smoke.sh").read_text()
    assert "materialize_ordered_shard.py" in text
    assert "SMOKE_ORDER_FILE" in text
    assert " train_single " in text
```

- [ ] **Step 2: Run and confirm failure**

```bash
pytest -v src/test/scripts/hypothesis/wrapper_contract_test.py
```

- [ ] **Step 3: Repair Skill-DAG wrapper**

Use the generated mix first:

```bash
MIX_FILE="$DATA_DIR/manifests/mixes/${MIX_NAME}.json"
if [[ ! -f "$MIX_FILE" ]]; then
  MIX_FILE="$ROOT/src/scripts/hypothesis/skill_dag/configs/${MIX_NAME}.json"
fi
export SMOKE_MODE=skill_dag
export SMOKE_DATA_DIR="$DATA_DIR"
export SMOKE_MIX_FILE="$MIX_FILE"
export WANDB_ENTITY="${WANDB_ENTITY:-eduLLM}"
export WANDB_PROJECT="${WANDB_PROJECT:-pretraining}"
export WANDB_GROUP="${WANDB_GROUP:-skill-dag-v1}"
SMOKE_RUN_MODE="${SMOKE_RUN_MODE:-dry-run}"
```

Always run the dry-run. Run `train_single` only when `SMOKE_RUN_MODE=train`; reject every other
value. Do not use `||`. Apply identical behavior to the PowerShell wrapper.

- [ ] **Step 4: Repair curriculum wrapper**

Call scoring, ordering, and materialization; export the resulting ordered shard. Use stable group
`curriculum-v1`. Correct README usage to separate pacing and metric arguments.

- [ ] **Step 5: Apply equivalent PowerShell behavior**

PowerShell must stop on failed dry-run, then explicitly run training. Keep argument and environment
names identical across platforms.

- [ ] **Step 6: Run tests and shell checks**

```bash
pytest -v src/test/scripts/hypothesis/wrapper_contract_test.py
bash -n src/scripts/hypothesis/skill_dag/run_smoke.sh
bash -n src/scripts/hypothesis/curriculum/run_smoke.sh
```

- [ ] **Step 7: Commit**

```bash
git add src/scripts/hypothesis src/test/scripts/hypothesis/wrapper_contract_test.py
git commit -m "fix: execute hypothesis smoke conditions"
```

---

### Task 5: Harden Data Preparation and Provenance

**Files:**
- Modify: `src/scripts/hypothesis/shared/prepare_smoke_dolma.py`
- Modify: `src/scripts/hypothesis/README.md`
- Modify: `src/scripts/hypothesis/CHANNEL_SMOKE.md`
- Test: `src/test/scripts/hypothesis/smoke_data_test.py`

**Interfaces:**
- Produces: nonempty validated domains, tokenizer identity, raw shard checksums.

- [ ] **Step 1: Add failing validation tests**

Require preparation validation to reject:

- A domain with zero documents.
- Missing tokenizer output.
- Null token paths.
- Empty token shards.
- Token IDs outside the padded vocabulary.
- Unpinned dataset or tokenizer revisions.

- [ ] **Step 2: Implement fail-fast preparation**

Remove the raw-only success path. If tokenizer load/tokenization fails, exit nonzero. Add per-shard
SHA-256 and padded vocabulary to `domain_index.json`.

Make immutable revisions required CLI arguments:

```python
ap.add_argument("--dataset-revision", required=True)
ap.add_argument("--tokenizer-revision", required=True)
```

Pass `revision=args.dataset_revision` to `load_dataset()`. Pass
`revision=args.tokenizer_revision, trust_remote_code=False` to `AutoTokenizer.from_pretrained()`.
Record both revisions in `domain_index.json` and `hypothesis_identity.json`.

Write generated natural mixtures using the same schema as committed configs:

```json
{
  "name": "natural",
  "weights": {
    "web": 0.4,
    "code": 0.2,
    "stem": 0.4
  }
}
```

Add an end-to-end test that feeds the generator's exact JSON into `load_mix_weights()`.

- [ ] **Step 3: Update documentation**

Document:

- Smoke pack is public/research-cleared.
- W&B secrets are environment-only.
- Exact dry-run/train sequence.
- W&B projects/groups.
- Smoke verifies plumbing, not hypothesis validity.

- [ ] **Step 4: Run tests**

```bash
pytest -v src/test/scripts/hypothesis/
```

- [ ] **Step 5: Commit**

```bash
git add src/scripts/hypothesis src/test/scripts/hypothesis
git commit -m "fix: validate hypothesis smoke data"
```

---

### Task 6: CPU Dry Runs and ORCD GPU Acceptance

**Files:**
- Modify: `src/scripts/hypothesis/README.md`

**Interfaces:**
- Produces: four W&B runs with verified data identities.

- [ ] **Step 1: Run all CPU unit tests**

```bash
pytest -v src/test/scripts/hypothesis/
make lint-check
make style-check
git diff --check
```

Expected: PASS.

- [ ] **Step 2: Prepare tiny Dolma data as a CPU Slurm job**

Run on Engaging `mit_normal`, not a login node. Verify manifests/shards and record their hashes.

- [ ] **Step 3: Dry-run all conditions**

```bash
SMOKE_STEPS=20 SMOKE_RUN_MODE=dry-run WANDB_PROJECT=pretraining \
  bash src/scripts/hypothesis/skill_dag/run_smoke.sh "$DATA_DIR" natural

SMOKE_STEPS=20 SMOKE_RUN_MODE=dry-run WANDB_PROJECT=pretraining \
  bash src/scripts/hypothesis/skill_dag/run_smoke.sh "$DATA_DIR" fixed_uniform

SMOKE_STEPS=20 SMOKE_RUN_MODE=dry-run WANDB_PROJECT=pretraining \
  bash src/scripts/hypothesis/curriculum/run_smoke.sh "$DATA_DIR" random compression_ratio

SMOKE_STEPS=20 SMOKE_RUN_MODE=dry-run WANDB_PROJECT=pretraining \
  bash src/scripts/hypothesis/curriculum/run_smoke.sh "$DATA_DIR" linear compression_ratio
```

During dry-run verification, disable W&B. Queue requests set `SMOKE_RUN_MODE=train` only inside a
GPU allocation and enable W&B there.

- [ ] **Step 4: Create queue requests through `/submit-edullm-job`**

Create four Issues with:

- Same reviewed commit.
- Same model and equal steps/tokens within each pair.
- Groups `skill-dag-v1` and `curriculum-v1`.
- Conditions as listed above.
- One L40S, 45-minute cap.

- [ ] **Step 5: Submit and monitor**

Each job must:

- Complete 20 steps.
- Produce finite loss.
- Save `config.json`, `metrics.json`, and `hypothesis_identity.json`.
- Appear in the correct W&B project/group.
- Have a data-artifact SHA distinct between conditions where expected.

- [ ] **Step 6: Record acceptance evidence**

Add non-secret Slurm IDs, W&B URLs, data hashes, runtimes, and failures to the hypothesis README.
Explicitly label results as smoke validation.

- [ ] **Step 7: Build and verify the shared Apptainer environment**

Create `src/Dockerfile.orcd`:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG SOURCE_SHA
COPY . /app/olmo-core
RUN pip install --no-deps /app/olmo-core
LABEL org.opencontainers.image.revision=${SOURCE_SHA}
WORKDIR /app/olmo-core
```

Create `src/scripts/orcd/build_apptainer.sh` that:

1. Resolves `ghcr.io/allenai/olmo-core:tch2100cu128-2025-11-25` to an immutable digest with
   `docker buildx imagetools inspect`.
2. Writes the full `repository@sha256:...` value to
   `config/edullm/orcd-base-image.txt`.
3. Requires a clean reviewed source SHA.
4. Builds `linux/amd64` with `src/Dockerfile.orcd`.
5. Pushes `ghcr.io/edu-llm/olmo-core-orcd:$SOURCE_SHA`.
6. Resolves the child digest and writes it to `config/edullm/orcd-image.txt`.

On Engaging:

```bash
singularity pull "edullm-${SOURCE_SHA}.sif" \
  "docker://$(cat config/edullm/orcd-image.txt)"
sha256sum "edullm-${SOURCE_SHA}.sif" > "edullm-${SOURCE_SHA}.sif.sha256"
singularity exec --nv "edullm-${SOURCE_SHA}.sif" nvidia-smi
```

Run the generic 20-step smoke inside the image with only reviewed code, data, and output
directories bound. Record the SIF SHA-256 in operator configuration. Update `edullm run` to use
`singularity exec --nv` when an image is configured; retain the virtualenv only for the pilot
operator until image acceptance passes.

Add tests that reject mutable image tags, missing source-revision labels, wrong SIF SHA-256, and
non-`linux/amd64` images.

- [ ] **Step 8: Enable the other operators after ORCD approval**

After ORCD confirms the shared project workflow, add the two reviewed operator GitHub/Slack
mappings to `config/edullm/operators.yaml`, enable them one at a time, run `edullm setup` on each
machine, install and verify the exact same SIF SHA-256, and submit two independent one-L40S
generic smokes. Confirm least-loaded assignment never exceeds two active GPUs per account.

- [ ] **Step 9: Commit**

```bash
git add src/scripts/hypothesis/README.md src/Dockerfile.orcd \
  src/scripts/orcd/build_apptainer.sh config/edullm/orcd-*-image.txt
git commit -m "docs: record hypothesis smoke validation"
```
