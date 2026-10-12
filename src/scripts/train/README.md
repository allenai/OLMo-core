## Ai2 internal training scripts

## Notice❗

The scripts in this folder use an internal module (`olmo_core.internal`), which is subject to breaking changes without notice,
and reference data paths that are only accessible to Ai2 employees, so they won't work out-of-the-box for external users.
Instead, see [`src/scripts/official/`](https://github.com/allenai/Olmo-core/tree/main/src/scripts/official) for public versions.

## Usage

Most of the scripts here have a consistent command-line API, but that's not guaranteed.
Generally if you run the script without any arguments it will print out some usage information.

---

Our own code does not live here. The canonical data → train → eval recipe is in
[`recipes/`](../../../recipes/README.md), and experiment code is in `research/<family>/`. Upstream
AI2 scripts stay at this level.
