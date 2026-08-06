---
name: submitting-a-run
description: >-
  Submits a training, evaluation or tokenization run to the eduLLM platform through the
  edullm CLI, reading refusal codes, cost and the approver from the machine-readable output
  rather than from prose. Use when the user asks to run, train, evaluate, sweep or submit
  something on the cluster, when a submission was refused and they want to know why, or when
  a run is in flight and they want its state.
---

# Submitting a run

The platform takes a commit, not a working tree. A run names a commit in a registered
repository, the container is built from that commit, and the record names what ran. Anything
uncommitted is not part of the run.

## The loop

```
- [ ] 1. Check, and read the JSON
- [ ] 2. Fix every refusal
- [ ] 3. Read what the shape you picked does about capacity
- [ ] 4. Check again until it is clean
- [ ] 5. Say what it costs and who has to release it
- [ ] 6. Submit
- [ ] 7. Follow it
```

### 1. Check, and read the JSON

```bash
edullm check --json --experiment <slug> --dataset <release-or-none>
```

`--experiment` is a lower-case hyphenated name for the question this run helps answer. It
registers nothing, so invent one. `--dataset` is a registered corpus, or the literal word
`none` where the run reads no corpus, which is what a smoke test, a tokenization or an
evaluation over existing checkpoints does. Absent and `none` are different answers and only
one of them is a statement.

**Read stdout on its own.** The first `check` in a repository with no `.edullm/run.yaml`
writes one and says so on stderr, so `edullm check --json 2>&1 | ...` turns that note into a
parse error on the one run where you least want one.

`check` reaches no network, so run it as often as you like. It costs a fraction of a second.

Branch on the exit code before reading anything. 0 stands, 1 refused on the merits, 2 the
tool could not be driven, 3 the platform could not be asked, 130 interrupted. Only 3 is worth
retrying.

Every document carries `format_version`, `edullm_version` and `verb`. A check carries these.

| Key | What it holds |
| --- | --- |
| `refused` | whether anything stopped it |
| `refusals` | a list of `{code, detail}` |
| `deferred` | checks a laptop cannot make, listed even when nothing is refused |
| `cost` | the factors and their product |
| `approval_class` | who has to release it |
| `manifest` | exactly what would be submitted, including the command and the commit |
| `history` | what runs of this shape have taken, with `said` as the sentence to quote |
| `config_directory` | the reviewed configuration this install carries |

### 2. Fix every refusal

Every entry is a `code` and a `detail`. **Match on `code`.** The detail is written for a
person and gets reworded. It names the field and usually the file.

| Code | What to do |
| --- | --- |
| `submitter_unknown` | `gh auth login`. Nothing can be priced until the roster can be asked about somebody |
| `no_experiment` | Pass `--experiment` with a lower-case hyphenated name. It registers nothing |
| `no_dataset` | Pass `--dataset`. Pass `none` where the run reads nothing. Absent and `none` are different answers |
| `experiment_not_a_slug` | Lower case, digits, single hyphens between words, none at either end |
| `team_is_ambiguous` | Pass `--team`. The `detail` lists the ones the roster puts this person on |
| `uncommitted_changes` | Commit or stash. The container is built from the last commit, so what would run is not what is on disk |
| `commit_not_pushed` | Push. A push to an `edullm/**` branch is what builds the image. If you just pushed, `git fetch` first |
| `unregistered_repository` | This codebase is not registered. Switch to the `registering-a-repository` skill |
| `unregistered_workload_profile` | The `detail` lists the registered ones. Pass `--workload` with one of those |
| `workload_profile_repository_mismatch` | That workload belongs to another repository. The `detail` lists this one's |
| `unregistered_dataset` | The `detail` lists what is registered. Do not invent a release id |
| `retired_dataset_release` | The corpus is registered and withdrawn. The `detail` names the version its owner calls current |
| `dataset_is_not_a_corpus` | This resolves to a tokenizer or another input rather than to something a run trains on |
| `unprovisioned_compute_profile` | The shape is priced and has no compute environment behind it, so no job on it can start. Pick another `--compute` |
| `process_per_device` | The command starts a different number of processes from the number of cards. Fix the launcher or pick a smaller `--compute` |
| `bfloat16_not_in_the_hardware` | The chosen card cannot do the dtype the command asks for. Pick another shape, or set the run to float32 |
| `checkpoint_path_not_in_command` | The workload promises a checkpoint a retry resumes from and the command never expands `$EDULLM_CHECKPOINT_DIR` |
| `retry_without_a_checkpoint_contract` | A retry on a workload that checkpoints nothing restarts from the beginning. Drop `--attempts` or pick a workload that checkpoints |

Anything else, read the `detail`. It was written to be acted on.

### 3. Read what the shape you picked does about capacity

Three things a clean check does not promise, in the order they cost the most.

**The dtype the code sets, rather than the dtype the command names.** The guard behind
`bfloat16_not_in_the_hardware` reads the text of the command. A trainer that fixes its
precision in code carries no bfloat16 token in argv, so the guard sees nothing and a card
without the format in hardware refuses the first kernel that needs it, after the run has been
priced, released, admitted and given a machine. **Write the dtype into the command**, which
turns a dead machine into a free refusal.

```bash
bash -lc 'python train.py train_module.dp_config.param_dtype=bfloat16'
```

**Whether EC2 will sell this account that machine.** Nothing in `check` reads it, and a job
whose shape never places sits in `RUNNABLE` with no error against it, which looks exactly like
being queued behind somebody. The verdict ships with the install, so read it before you pick.

```bash
CONFIG=$(edullm check --json --experiment a-first-look --dataset none 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["config_directory"])')
grep -A1 'profile: gpu-8xa100' "$CONFIG/capacity.yaml"
```

| What `places` says | What it means |
| --- | --- |
| `reliably` | the machine starts |
| `after_a_wait` | it arrives, and the entry beside it says what the wait has been |
| `unreliably` | it may never arrive |

Several shapes the catalog still offers read `unreliably`. Read the shape rather than assuming
the catalog lists only machines you can get.

**The image checks under `deferred`.** They need the container registry, which needs a
credential this tool does not hold, so they are made at submit time instead. `check` lists
them on a clean run for that reason.

### 4. Check again until it is clean

Do not skip to submit with refusals outstanding, and do not reach for `--force`. Every refusal
`--force` skips is one admission makes again from inside AWS, so it buys a queue wait rather
than an outcome.

### 5. Say what it costs and who has to release it

`cost` carries the factors and their product.

```
maximum_compute_cost_usd = hourly_rate_usd x nodes x maximum_runtime_hours x maximum_attempts x cells
```

**Report `maximum_compute_cost_usd`.** A fan-out multiplies and `cells` is where it
multiplies, so quoting the hourly rate, or the cost of a single cell, to somebody about to
approve the lot misleads them about the size of what they are approving.

`approval_class` says who releases it. **Read the value.** Do not work out from a figure you
remember which class a run lands in.

| `approval_class` | Who releases the run |
| --- | --- |
| `automatic` | nobody. It starts as soon as admission accepts it |
| `routine` | a team lead, who has to open the run page and approve |
| `exception` | nobody reaches this today. It is machinery kept for reserved-capacity purchases, and no submission classifies into it |

A fan-out goes to a person whatever it costs, and so does anything the reviewed configuration
prices at or above its automatic bound. Those rules and the figure live in configuration that
has been re-cut before, which is why the answer is a field to read rather than a sum to do.

Say both before you submit. A submission left at a gate overnight looks queued from the
outside, and the person who thinks their job is running finds out in the morning.

### 6. Submit

```bash
edullm submit --experiment <slug> --dataset <release-or-none>
```

It prints the workflow run URL, then the run id on a line of its own, then whether the run was
released automatically or is waiting at an approval gate. **Keep the run id.**

If it says the submission is waiting, a person has to tap. Nothing you can run releases it.

### 7. Follow it

```bash
edullm status --json <run-id>
```

This answers from GitHub and dispatches nothing, so it is free and you may poll it. Read
`admitted` and `needs_a_dispatch`. On a run still at a gate, `gate` and `reviewers` name who
is being waited on and `you_can_release` says whether the submitter is one of them.

When `needs_a_dispatch` is true the run has reached AWS and the rest of the answer costs a
runner. Run the same verb without `--json`, or `edullm logs <run-id>` for what it printed.
Both are slow by construction because a workflow has to start. Neither belongs in a loop, and
neither has a `--json`, because what they print is a section of a job log.

`edullm cancel <run-id> --reason "<why>"` stops an admitted run, and the reason is what the
run's history records instead of a failure.

## The exploration lane is not this

`edullm run` ships this working tree to a machine of your own and streams back the output of
the command after a bare `--`. `edullm shell` gives you a terminal on that machine. Nothing on
that route is checked against the registry, priced, approved or recorded, so what comes off it
is a thing somebody saw rather than a result anybody can cite. It is the wrong answer to a
refused submission.

## Never

- Never call AWS directly. No `boto3`, no `aws` CLI. The binary is the interface.
- Never parse the human output when `--json` exists on that verb.
- Never quote a price, a runtime bound, a ceiling or who approves from memory or from this
  file. `edullm check --json` prints them out of the reviewed configuration.
- Never pass `--force`.
- Never report the cost of one cell of a fan-out as the cost of the submission.
