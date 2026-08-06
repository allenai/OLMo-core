# AGENTS.md

Read [`CLAUDE.md`](CLAUDE.md). It is the guidance for this repository and it applies to
every coding agent, not only Claude Code. This file exists because Cursor and Codex read
`AGENTS.md` and would otherwise find nothing.

Two things worth knowing before you read it:

- **GPU runs go through the eduLLM platform's `edullm` CLI, not Beaker.** The `ai2/*`
  clusters named throughout this repository belong to AI2 and cannot be reached from here.
  See "Running on GPUs" in `CLAUDE.md`.
- **`.edullm/train_on_corpus.py` trains in bfloat16 by default, and the platform cannot see
  a dtype that is set in code.** Write it into the command. A T4 has no bfloat16 in the
  hardware, and a command that does not name the dtype is accepted onto one.

<!-- edullm:begin -->
<!-- Managed by edu-llm/platform. Edit skills/agents-md-block.md there and re-run
     tools/distribute_agent_layer.py; an edit made here is reverted and, until it is,
     tests/test_agent_layer_is_distributed.py is red. Text outside the markers is
     this repository's own and is never touched. -->

## Running anything on a GPU: use `edullm`, never AWS

This codebase is registered with the eduLLM platform. `edullm` is the only supported way to
reach the cluster from a laptop, and it holds no cloud credential of its own: every AWS
credential lives in a workflow whose trust policy pins it to one file on `main`.

**Do not write a script that calls AWS.** No `boto3`, no `aws` CLI, no `curl` at an AWS
endpoint. For the people here who hold no AWS role that fails, and for the few who do it
succeeds and leaves no run anybody can cite, which is the worse of the two.

```bash
uv tool install --force git+https://github.com/edu-llm/platform
edullm --version
```

Unpinned on purpose, and **re-running that line is the upgrade**. Do not reach for `pip` or
`pipx`, and do not reach for `uv tool upgrade`: what it does depends on how the tool was
installed, so from a release note's pinned line it answers `Nothing to upgrade` and exits 0
however far behind the install is. `uv tool install edullm` is the other near miss and uv
answers `not found in the package registry`, because nothing is published to an index under
that name; the line above installs from git.

If this machine installed before the distribution was renamed, run `uv tool uninstall
edullm-platform` **before** the install line and not after. Both own the same `edullm`
executable and uv deletes it along with the old entry, which leaves a healthy-looking
`uv tool list` and no command.

It needs `gh` logged in and a clone with an `origin` remote, and nothing else — no AWS
profile, no SSO session and no VPN, for anything on this path.

| Verb | What it does |
| --- | --- |
| `edullm check` | Prices a submission from this working tree and lists every refusal. Reaches no network. |
| `edullm submit` | Runs those checks and dispatches the submission workflow. |
| `edullm status` | Names your recent submissions, or describes one run. |
| `edullm logs` | The last lines one run printed. |
| `edullm cancel` | Stops one admitted run, with a reason that goes on the record. |
| `edullm add` | Teaches the platform about a repository, dataset, shape, model or person. |
| `edullm ask` | Files one ask for something you need yourself. |
| `edullm run` / `edullm shell` | Ships this tree to a machine of your own. Ungated, and no run anybody can cite. |

`edullm <verb> --help` prints what that verb takes. The last two are the exploration route
and not the submission path: nothing on them is checked, priced, approved or recorded.

**Start with `edullm check --json`.** It costs a fraction of a second, reaches no network and
lists every refusal at once. Read the JSON on stdout rather than the prose, and **match on
`code`** — the `detail` beside it is written for a person and gets reworded. Exit 0 stands,
1 is refused on the merits, 2 means the command or the install is wrong, 3 means the platform
could not be asked and is the only one worth retrying.

**Never quote a price, a runtime bound, a cost ceiling or who has to approve something from
memory or from a document, this one included.** Those live in reviewed configuration that
changes without anybody being told. Run `edullm check --json` and read `cost` and
`approval_class` out of the output.

Two skills carry the detail and your agent should reach for them by name rather than working
it out here: **submitting-a-run** for anything that runs, and **registering-a-repository**
when `check` refuses with `unregistered_repository`.

Also never: pass `--force` to get past a refusal, edit `.edullm/run.yaml` to silence a
refusal without reading what it says, or commit a secret into this repository — the image is
built from the commit.
<!-- edullm:end -->
