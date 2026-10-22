"""
Mirror local dirs to/from the olmo3_cptdrop S3 hub (s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/).
boto3 only (no aws CLI on the Berkeley nodes); files whose size already matches are skipped.

    python s3_sync.py up   /data/prasann/olmo3_cptdrop/data/train32k data/train32k
    python s3_sync.py down /data/prasann/olmo3_cptdrop/data/train32k data/train32k
"""
from __future__ import annotations

import os
import sys
import time

import boto3
from boto3.s3.transfer import TransferConfig

BUCKET = "ai2-llm"
ROOT = "checkpoints/prasanns/_transfer/olmo3_cptdrop"
CFG = TransferConfig(multipart_chunksize=128 << 20, max_concurrency=16)


def main():
    direction, local, rel = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3].strip("/")
    s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "S3")).client("s3")
    prefix = f"{ROOT}/{rel}/"
    remote = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            remote[o["Key"][len(prefix):]] = o["Size"]
    t0, n, nbytes = time.time(), 0, 0
    if direction == "up":
        for dp, _, fs in os.walk(local):
            for f in sorted(fs):
                p = os.path.join(dp, f)
                k = os.path.relpath(p, local)
                size = os.path.getsize(p)
                if remote.get(k) == size:
                    continue
                s3.upload_file(p, BUCKET, prefix + k, Config=CFG)
                n, nbytes = n + 1, nbytes + size
                print(f"up {k} {size/1e9:.2f} GB ({time.time()-t0:.0f}s)", flush=True)
    elif direction == "down":
        for k, size in sorted(remote.items()):
            p = os.path.join(local, k)
            if os.path.exists(p) and os.path.getsize(p) == size:
                continue
            os.makedirs(os.path.dirname(p), exist_ok=True)
            s3.download_file(BUCKET, prefix + k, p, Config=CFG)
            n, nbytes = n + 1, nbytes + size
            print(f"down {k} {size/1e9:.2f} GB ({time.time()-t0:.0f}s)", flush=True)
    else:
        raise SystemExit("direction must be up or down")
    print(f"DONE {direction} s3://{BUCKET}/{prefix}: {n} files, {nbytes/1e9:.2f} GB, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
