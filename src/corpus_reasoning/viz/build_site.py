"""
Orchestrator — run the full visualization pipeline in one process.

    extract_data        sample task examples across context ladders
    collect_experiments gather CPT-mix experiment configs + results
    render              emit the self-contained index.html

Usage:
    python viz/build_site.py                # build outputs/index.html
    python viz/build_site.py --update-demo  # also refresh the committed demo/index.html
"""

import argparse
import os
import shutil
import subprocess
import sys

# Make the modules importable whether invoked as `python viz/build_site.py`
# or `python -m viz.build_site`.
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
import collect_docchunk  # noqa: E402
import collect_experiments  # noqa: E402
import collect_headline  # noqa: E402
import collect_mixing  # noqa: E402
import config  # noqa: E402
import extract_data  # noqa: E402
import render  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build the corpus-reasoning visualization site")
    ap.add_argument("--update-demo", action="store_true",
                    help="copy the rendered site to the committed demo/index.html")
    ap.add_argument("--publish", action="store_true",
                    help="after building, S3-sync outputs/ to VIZ_S3_DEST")
    ap.add_argument("--deploy", action="store_true",
                    help="after building, deploy index.html to Cloudflare Pages")
    args = ap.parse_args()

    print("=== [1/3] extract_data ===")
    extract_data.main()
    print("=== [2/3] collect_experiments ===")
    collect_experiments.main()
    print("=== [2b/3] collect_mixing ===")
    collect_mixing.main()
    print("=== [2c/3] collect_docchunk ===")
    collect_docchunk.main()
    print("=== [2d/3] collect_headline ===")
    collect_headline.main()
    print("=== [3/3] render ===")
    render.main()

    if args.update_demo:
        os.makedirs(config.DEMO_DIR, exist_ok=True)
        shutil.copyfile(config.SITE_HTML, config.DEMO_HTML)
        print(f"=== updated demo: {config.DEMO_HTML} ===")

    if args.publish:
        print("=== publish: S3 sync ===")
        push = os.path.join(config.VIZ_DIR, "push_outputs.sh")
        rc = subprocess.run(["bash", push]).returncode
        if rc != 0:
            print("WARNING: publish failed (is VIZ_S3_DEST set + aws creds available?)")

    if args.deploy:
        print("=== deploy: Cloudflare Pages ===")
        deploy = os.path.join(config.VIZ_DIR, "deploy_cloudflare.sh")
        rc = subprocess.run(["bash", deploy]).returncode
        if rc != 0:
            print("WARNING: deploy failed (is wrangler authed? CLOUDFLARE_API_TOKEN / wrangler login)")

    print(f"\nDone. Open: {config.SITE_HTML}")


if __name__ == "__main__":
    main()
