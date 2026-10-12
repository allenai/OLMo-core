#!/usr/bin/env python3
"""
Walk one or more roots ONCE and write a compact storage index (JSON) that
`storage_report.py` can read in milliseconds. Read-only: never deletes, moves or modifies.

Run it where the disk is LOCAL (a slurm job on the node for /data, a Beaker job for weka, a
niced background process on the login node for /accounts and /scratch) — walking over NFS or
/net is exactly the hours-long D-state trap this split exists to avoid.

    python3 storage_index.py --label horton --root /data/prasann --out /data/prasann/.storage_index.json

Records, per directory down to --max-depth with >= --min-dir-gb on disk: bytes on disk,
file count, newest FILE mtime anywhere below it (the "last touched" signal; directory
mtimes are ignored because deleting children bumps them), and flags:
  redundant_stepN   a stepN checkpoint whose run also has a model-only `model_and_optim/.metadata`,
                    or a stepN that is not the run's highest step (keep-final would drop it)
Plus the largest individual files, df / zfs info for the filesystem, and per-root totals.
Progress goes to <out>.progress every ~10 s; the JSON is written atomically at the end.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import socket
import stat
import subprocess
import sys
import time

STEP_RE = re.compile(r"^step(\d+)$")


class Walker:
    def __init__(self, a: argparse.Namespace):
        self.min_dir = int(a.min_dir_gb * 2**30)
        self.min_file = int(a.min_file_gb * 2**30)
        self.max_depth = a.max_depth
        self.top_files = a.top_files
        self.dirs: list[list] = []
        self.files: list[tuple] = []  # min-heap of (bytes, path, mtime)
        self.seen_inodes: set[tuple[int, int]] = set()
        self.n_files = self.n_dirs = self.errors = 0
        self.progress_path = a.out + ".progress"
        self.t0 = self.last_progress = time.time()
        self.root_dev = 0

    def progress(self, where: str, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_progress < 10:
            return
        self.last_progress = now
        msg = (f"{time.strftime('%H:%M:%S')} {now - self.t0:7.0f}s  dirs={self.n_dirs:,} "
               f"files={self.n_files:,} errors={self.errors}  at {where[:160]}\n")
        try:
            with open(self.progress_path, "a") as fh:
                fh.write(msg)
        except OSError:
            pass

    def walk(self, path: str, depth: int) -> tuple[int, int, float]:
        """Returns (bytes_on_disk, n_files, newest_mtime) for the subtree."""
        self.n_dirs += 1
        self.progress(path)
        total, nfiles, newest = 0, 0, 0.0
        children: dict[str, tuple[int, int, float]] = {}
        try:
            it = os.scandir(path)
        except OSError:
            self.errors += 1
            return 0, 0, 0.0
        with it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    self.errors += 1
                    continue
                if stat.S_ISDIR(st.st_mode):
                    if st.st_dev != self.root_dev:  # stay on one filesystem (like du -x)
                        continue
                    sub = self.walk(e.path, depth + 1)
                    children[e.name] = sub
                    total += sub[0]
                    nfiles += sub[1]
                    # Directory mtimes are NOT activity: deleting a child (e.g. a cleanup)
                    # bumps them. "Last touched" is the newest FILE mtime below.
                    newest = max(newest, sub[2])
                    continue
                if st.st_nlink > 1:
                    key = (st.st_dev, st.st_ino)
                    if key in self.seen_inodes:
                        continue
                    self.seen_inodes.add(key)
                size = st.st_blocks * 512
                total += size
                nfiles += 1
                self.n_files += 1
                newest = max(newest, st.st_mtime)
                if size >= self.min_file:
                    item = (size, e.path, st.st_mtime)
                    if len(self.files) < self.top_files:
                        heapq.heappush(self.files, item)
                    elif size > self.files[0][0]:
                        heapq.heapreplace(self.files, item)

        # Redundant checkpoint detection at the run level (this dir's children).
        steps = {n: int(m.group(1)) for n in children if (m := STEP_RE.match(n))}
        if steps:
            meta = os.path.join(path, "model_and_optim", ".metadata")
            try:
                has_modelonly = os.path.getsize(meta) > 0
            except OSError:
                has_modelonly = False
            top = max(steps.values())
            for n, k in steps.items():
                b, f, t = children[n]
                if b < self.min_dir:
                    continue
                why = ("run has model-only model_and_optim" if has_modelonly else
                       f"not the highest step (step{top} exists)" if k != top else "")
                if why and depth + 1 <= self.max_depth + 1:
                    self.dirs.append([os.path.join(path, n), b, f, t, depth + 1,
                                      "redundant_stepN: " + why])

        if depth <= self.max_depth and total >= self.min_dir:
            self.dirs.append([path, total, nfiles, newest, depth, ""])
        return total, nfiles, newest


def run_quiet(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=socket.gethostname())
    ap.add_argument("--min-dir-gb", type=float, default=1.0)
    ap.add_argument("--min-file-gb", type=float, default=1.0)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--top-files", type=int, default=300)
    a = ap.parse_args()

    w = Walker(a)
    try:
        os.remove(w.progress_path)
    except OSError:
        pass
    totals = {}
    for root in a.root:
        root = os.path.abspath(root)
        try:
            w.root_dev = os.stat(root).st_dev
        except OSError as e:
            totals[root] = dict(error=str(e))
            continue
        b, f, t = w.walk(root, 0)
        totals[root] = dict(bytes=b, files=f, newest=t)
    w.progress("done", force=True)

    df = {}
    for root in a.root:
        out = run_quiet(["df", "-P", "-B1", root]).splitlines()
        if len(out) >= 2:
            p = out[-1].split()
            df[root] = dict(fs=p[0], size=int(p[1]), used=int(p[2]), avail=int(p[3]),
                            mount=p[-1])
    zfs = {}
    zl = run_quiet(["zfs", "list", "-Hp", "-o", "name,used,avail,usedbysnapshots"])
    if zl:
        zfs["datasets"] = [dict(zip(["name", "used", "avail", "usedbysnap"], ln.split("\t")))
                           for ln in zl.splitlines()]
    zu = run_quiet(["zfs", "userspace", "-Hp", "-o", "name,used", "pool0/data"])
    if zu:
        zfs["userspace"] = {ln.split("\t")[0]: int(ln.split("\t")[1])
                            for ln in zu.splitlines() if "\t" in ln}

    res = dict(label=a.label, host=socket.gethostname(), roots=a.root,
               generated=time.time(), elapsed_s=round(time.time() - w.t0, 1),
               n_files=w.n_files, n_dirs=w.n_dirs, errors=w.errors,
               min_dir_bytes=w.min_dir, max_depth=a.max_depth,
               totals=totals, df=df, zfs=zfs,
               dirs=sorted(w.dirs, key=lambda d: -d[1]),
               files=[[p, b, t] for b, p, t in sorted(w.files, reverse=True)])
    tmp = a.out + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(res, fh, separators=(",", ":"))
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {w.n_files:,} files, {w.n_dirs:,} dirs, "
          f"{len(res['dirs'])} indexed dirs in {res['elapsed_s']}s", file=sys.stderr)


if __name__ == "__main__":
    main()
