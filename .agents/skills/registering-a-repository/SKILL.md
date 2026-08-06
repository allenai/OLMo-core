---
name: registering-a-repository
description: >-
  Registers a research codebase with the eduLLM platform so it can build an image and accept
  runs, by writing the three files the repository needs and opening the configuration pull
  request through edullm add repository. Use when edullm check refuses with
  unregistered_repository, when a codebase has no .edullm directory, or when the user asks to
  get a new repository onto the platform.
---

# Registering a repository

Registration is two halves and they are easy to confuse. **The platform half** edits reviewed
configuration files and is opened as a pull request by a workflow. **The repository half** is
three files in the codebase being registered, and nothing writes them for you. A repository
whose pull request merged and whose three files are missing is registered and can never build
an image, which has already happened once here.

Do the repository half first. A pull request opened against a codebase with no Dockerfile
declares a path that points at nothing.

## The loop

```
- [ ] 1. Confirm it is not already registered
- [ ] 2. Resolve the base image against what is approved
- [ ] 3. Write .edullm/Dockerfile
- [ ] 4. Write the build-caller workflow
- [ ] 5. Write a first .edullm/run.yaml
- [ ] 6. Commit and push to an edullm/** branch
- [ ] 7. Open the configuration pull request
- [ ] 8. Say what happens next
```

### 1. Confirm it is not already registered

```bash
edullm check --json --experiment a-first-look --dataset none
```

A `refusals` entry with code `unregistered_repository` means it is not. Its `detail` lists
what is registered. Any other answer means it already is, and you are in the wrong skill.

### 2. Resolve the base image against what is approved

This is the one question a reviewer has to answer, so answer it before asking.

Read `config/repositories.yaml` in the platform repository and list the base images existing
registrations already carry. Resolve this codebase's dependency set against them.

- If one of the approved bases satisfies the dependency set, use it. Say which, and say that
  it is one already reviewed.
- If none does, name the single pin that forces a new base. A new base is a second thing to
  review, scan and re-pin, so it needs a reason and the reason is that pin.

Do not pick a base because the project's own Dockerfile uses it. That is not a reviewed
answer.

### 3. Write `.edullm/Dockerfile`

It builds from the base you resolved, installs the dependency set, and does nothing at
runtime. The command a run executes comes from the submission, not from the image.

Keep it minimal. Every layer is rebuilt on every push to an `edullm/**` branch and the build
cache is one of two levers on a bill that is not small.

### 4. Write the build-caller workflow

A workflow in the research repository that calls the platform's reusable build. **Check what
it fires on.** A caller that fires only on `edullm/**` pushes and manual dispatch never fires
for a branch named anything else, and that is exactly how a registered repository ends up with
zero images while looking correct. If the work lives on a branch that is not `edullm/**`, say
so, and say the two ways out, which are renaming the branch or dispatching the caller by hand.

**Then set `AWS_ECR_PUBLISHER_ROLE_ARN` as a repository variable, which the workflow you just
wrote reads and which nothing gives you.** Settings, then Secrets and variables, then Actions,
then Variables, on the research repository itself. `gh variable set` does it from a terminal,
given the name, the value and the repository.

The ARN is the one `infra/README.md` records for `sbsandbox-intern-edullm-ecr-publisher`. It
is set per repository, by hand, in each: **there is no organization variable behind it**, so
the repositories that already have one tell you nothing about this one, and registering a
repository does not create it.

Until 2026-08-06 this step was in no document at all. `edullm-p1` read as fully registered and
published nothing for days because of exactly that. It is not a step the platform can check for
you either — a token scoped to `edu-llm/platform` is refused by every other repository's
variables endpoint — so the check lives in the reusable build, whose first step refuses an
empty value with `publisher_role_arn_is_empty` and the variable's name. If you see that, this
is the step you skipped.

### 5. Write a first `.edullm/run.yaml`

It holds what is a property of the code, which is the command, the workload profile and a
suggested machine. Everything else is supplied at submit time.

`edullm check` writes this file itself once the repository is registered, so the version you
write here is a placeholder that gets replaced. Write it anyway. It is what makes the pull
request reviewable.

### 6. Commit and push to an `edullm/**` branch

```bash
git switch -c edullm/register
git add .edullm/
git commit -m "Add the platform's build inputs"
git push -u origin edullm/register
```

The push is what builds the first image.

### 7. Open the configuration pull request

```bash
edullm add repository --reason "<why this needs a repository of its own>"
```

`--reason` has no default and it is the only part a reviewer cannot derive. Answer why this
needs a repository of its own rather than a workload profile in one that is already
registered.

The command prints the workflow run page. The pull request appears there.

### 8. Say what happens next

Tell the user plainly. The pull request has to be merged by the platform owner and then
deployed, and **nothing is registered until both have happened.** A merged configuration
change does nothing in the account until somebody deploys, and that has already cost one
incident. `edullm check` keeps refusing this repository until then.

## Never

- Never edit the platform's configuration files directly. The workflow edits five of them and
  runs a verification; a hand edit skips it.
- Never invent a base image. Resolve against `config/repositories.yaml`.
- Never claim the repository is registered because the pull request is open.
