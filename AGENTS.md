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
