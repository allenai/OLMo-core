# Corpus-Reasoning Visualization Site

A small pipeline that renders a **self-contained interactive HTML site** for this
project — no server, no build step, no external dependencies: open
`outputs/index.html` (or the committed `demo/index.html`) straight in a browser.

It has three views:

- **Overview** — suite stats + the task-complexity (CTC) legend.
- **Data Explorer** — every task in the suite, with real examples sampled across
  its context-length / item-count ladder (query, documents, gold answer, metadata).
- **Experiments** — the CPT data-mixing runs (does mixing continued-pretraining
  text back into SFT recover long-context / RULER ability?), with each run's
  description, config, SFT/CPT token mix, and a RULER-by-context-length results table.

The format follows the [`allenai/EMO`](https://github.com/allenai/EMO) cluster-explorer
pattern: a staged pipeline that emits one self-contained HTML with all data
embedded as JS consts, plus S3 sync scripts to share the artifact.

## Pipeline

```
extract_data.py        →  outputs/data_examples.json   (Data Explorer source)
collect_experiments.py →  outputs/experiments.json     (Experiments source)
render.py              →  outputs/index.html           (the site)
```

`build_site.py` runs all three.

## Quick start

```bash
# Build outputs/index.html from live data + experiment configs
bash viz/run.sh

# Also refresh the committed snapshot (demo/index.html)
bash viz/run.sh --update-demo

# Open it
xdg-open viz/outputs/index.html      # or just scp it to your laptop
```

## Where it reads from (all overridable)

| Env var | Default | Used for |
|---------|---------|----------|
| `CR_DATA_ROOT` | `/scratch/users/prasann/corpus-reasoning/data` | the task `*.jsonl` files (gitignored, on scratch) |
| `OLMO_CORE_ROOT` | the OLMo-core checkout that owns this submodule | the SFT/CPT-mix training scripts |
| `VIZ_OUT_DIR` | `viz/outputs` | where artifacts are written |

Because the data lives on scratch/weka (not in git), the pipeline reads it **by
path** rather than vendoring it — the same decoupling EMO uses for `claude_outputs`.
This is why `corpus-reasoning` works fine as a submodule of OLMo-core: code is
shared via git, data via path.

## Customizing

- **Add/remove tasks** — edit `TASK_MANIFEST` in `extract_data.py` (each entry is a
  glob + a regex that pulls the ladder value from the filename).
- **Add experiments** — add an entry to `EXPERIMENTS` in `collect_experiments.py`
  (it parses the script's `# Question this answers:` comment as the description and
  reads `CPT_FRAC` / `SEQUENCE_LENGTH` / … automatically).
- **Fill in results** — edit `results.json` as runs finish. `null` cells render as
  "—" (pending); live numbers are in the wandb project `memory-networks`.
- **Restyle** — all CSS/JS is in the `TEMPLATE` string in `render.py`.

## Publishing the website (Cloudflare Pages)

The site is a single self-contained `index.html`, so Cloudflare Pages serves it
directly. One-time account setup:

1. Create a free Cloudflare account.
2. Authenticate `wrangler` (run via `npx`, no global install needed), either:
   - headless / CI: `export CLOUDFLARE_API_TOKEN=...` and `export CLOUDFLARE_ACCOUNT_ID=...`
     (the token needs the **Cloudflare Pages: Edit** permission), or
   - interactively, once: `npx wrangler login`
3. (optional) `export CF_PAGES_PROJECT=corpus-reasoning-viz` to name the project.

Then deploy:

```bash
bash viz/run.sh --deploy      # build + deploy in one step
# or separately:
bash viz/run.sh
bash viz/deploy_cloudflare.sh  # -> https://<project>.pages.dev
```

The first deploy creates the Pages project and prints the public `*.pages.dev`
URL; subsequent deploys update it. Only `index.html` is published (the JSON
sources stay local). From OLMo-core the same works via `bash viz.sh --deploy`.

## Sharing raw artifacts via S3 / R2 (optional)

For artifact storage (not serving), sync `outputs/` to S3 — or to Cloudflare R2,
which is S3-compatible (`aws s3 sync --endpoint-url https://<acct>.r2.cloudflarestorage.com ...`):

```bash
export VIZ_S3_DEST=s3://<your-bucket>/<prefix>/corpus_reasoning_viz
bash viz/push_outputs.sh           # outputs/ -> S3/R2
bash viz/pull_outputs.sh           # S3/R2 -> outputs/ (no --delete)
```

## Files

```
viz/
├── README.md
├── config.py              # central paths (env-overridable)
├── extract_data.py        # stage 1: Data Explorer source
├── collect_experiments.py # stage 2: Experiments source
├── render.py              # stage 3: self-contained HTML
├── build_site.py          # orchestrator
├── run.sh                 # thin wrapper
├── results.json           # curated RULER results (edit by hand)
├── deploy_cloudflare.sh   # deploy index.html to Cloudflare Pages
├── push_outputs.sh        # S3/R2 sync (push)
├── pull_outputs.sh        # S3/R2 sync (pull)
├── demo/index.html        # committed snapshot of the rendered site
└── outputs/               # gitignored pipeline artifacts
```
