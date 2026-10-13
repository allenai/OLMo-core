"""``ops guard``: stall-watch wrapper for any long command.

    ops guard [--first-output 120] [--stall 300] [--log PATH] [--kill] [--name N] -- CMD ...

Streams the child's merged stdout/stderr (tee'd to ``--log``), and when the child prints nothing for
``--first-output`` s at the start or ``--stall`` s later, prints a loud ``[ops guard] STALL`` block that
diagnoses the child's process tree from /proc (state, wchan, CPU over a 2 s sample, thread wchans,
open files on NFS, py-spy dump when available) with the same classifier as ``ops stall``. It then keeps
waiting (the next report comes after twice the interval), or with ``--kill`` kills the process group
and exits 124. Sets PYTHONUNBUFFERED=1 / PYTHONFAULTHANDLER=1 for the child. Polls once a second.
"""
import glob
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from . import jobs

LOGIN_HOSTS = {"radagast"}
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


# ------------------------------------------------------------------------------------------------
# NFS detection


def _nfs_mounts():
    out = []
    try:
        with open("/proc/mounts") as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 3 and f[2].startswith("nfs"):
                    out.append(f[1])
    except OSError:
        pass
    return sorted(set(out), key=len, reverse=True)


def is_nfs(path, mounts=None):
    """True if ``path`` is on NFS: known NFS prefixes, or under an nfs mount in /proc/mounts."""
    if not path:
        return False
    p = os.path.abspath(str(path))
    if jobs.is_nfs_path(p + "/"):
        return True
    for m in (mounts if mounts is not None else _nfs_mounts()):
        if p == m or p.startswith(m.rstrip("/") + "/"):
            return True
    return False


def start_warnings(cmd, log=None, cwd=None, host=None, mounts=None):
    """Warnings printed at start when running on a compute node with NFS paths (pure-ish)."""
    host = (host or socket.gethostname()).split(".")[0]
    if host in LOGIN_HOSTS:
        return []
    warns = []
    cwd = cwd or os.getcwd()
    if log and is_nfs(log, mounts):
        warns.append(f"--log {log} is on NFS: a log written from a compute node can wedge in "
                     "nfs_wait_bit_killable; use /data/prasann/...")
    if is_nfs(cwd, mounts):
        warns.append(f"cwd {cwd} is on NFS: relative outputs/caches will land on NFS; cd to /data/... for job I/O")
    exe = cmd[0] if cmd else ""
    resolved = shutil.which(exe) or exe
    interp = resolved
    try:
        if resolved and os.path.isfile(resolved):
            with open(resolved, "rb") as fh:
                first = fh.readline(200)
            if first.startswith(b"#!"):
                interp = first[2:].decode(errors="replace").split()[0]
    except OSError:
        pass
    for p in {resolved, interp}:
        if p and re.search(r"python|torchrun|vllm", os.path.basename(p)) and is_nfs(p, mounts):
            warns.append(f"interpreter {p} is on NFS: imports page GBs over ~5 MB/s and look like a hang; "
                         "use the node-local env under /data/prasann/...")
            break
    return warns


# ------------------------------------------------------------------------------------------------
# /proc inspection of the child tree


def _read(path, default=""):
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return default


def proc_tree(root):
    """PIDs of ``root`` and all its descendants (from /proc/*/stat ppid)."""
    children = {}
    for d in glob.glob("/proc/[0-9]*"):
        st = jobs.parse_proc_stat(_read(d + "/stat"))
        if st:
            children.setdefault(st["ppid"], []).append(st["pid"])
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.append(p)
        stack.extend(children.get(p, []))
    return [p for p in out if os.path.exists(f"/proc/{p}")]


def snapshot_proc(pid):
    st = jobs.parse_proc_stat(_read(f"/proc/{pid}/stat"))
    if not st:
        return None
    threads = {}
    for t in glob.glob(f"/proc/{pid}/task/*"):
        w = _read(t + "/wchan").strip() or "-"
        ts = jobs.parse_proc_stat(_read(t + "/stat"))
        key = w + ("/" + ts["state"] if ts else "")
        threads[key] = threads.get(key, 0) + 1
    nfs_fds = {}
    mounts = _nfs_mounts()
    for fd in glob.glob(f"/proc/{pid}/fd/*")[:2000]:
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("/") and is_nfs(target, mounts):
            nfs_fds[target] = nfs_fds.get(target, 0) + 1
    env = {}
    for kv in _read(f"/proc/{pid}/environ").split("\0"):
        k, _, v = kv.partition("=")
        if k in ("HOME", "TMPDIR", "TRITON_CACHE_DIR", "FLASHINFER_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME"):
            env[k] = v
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        exe = None
    return {"pid": pid, "ppid": st["ppid"], "stat": st["state"], "wchan": _read(f"/proc/{pid}/wchan").strip(),
            "ticks": st["ticks"], "args": _read(f"/proc/{pid}/cmdline").replace("\0", " ").strip()[:300],
            "exe": exe, "threads": threads, "nfs_fds": sorted(nfs_fds, key=lambda k: -nfs_fds[k])[:6], "env": env}


def find_py_spy():
    p = shutil.which("py-spy")
    if p:
        return p
    for pat in ("/data/prasann/conda/envs/*/bin/py-spy", "/data/prasann/*/bin/py-spy"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def diagnose(root, silent_s, sample=2.0, pyspy=True, silent_threshold=None):
    """Build an ``ops stall``-style evidence dict for the local process tree rooted at ``root``."""
    pids = proc_tree(root)
    first = {p: snapshot_proc(p) for p in pids}
    time.sleep(sample)
    procs = []
    for p, s in first.items():
        if not s:
            continue
        st = jobs.parse_proc_stat(_read(f"/proc/{p}/stat"))
        s["cpu"] = (100.0 * (st["ticks"] - s["ticks"]) / CLK_TCK / sample) if st else None
        if st:
            s["stat"] = st["state"]
            s["wchan"] = _read(f"/proc/{p}/wchan").strip()
        procs.append(s)
    ev = {"state": "RUNNING", "procs": procs, "log_age_s": silent_s, "elapsed_s": None, "gpus": [],
          "silent_threshold": silent_threshold if silent_threshold is not None else silent_s,
          "login": socket.gethostname().split(".")[0] in LOGIN_HOSTS}
    if shutil.which("nvidia-smi"):
        rc, out, _ = jobs.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                               "--format=csv,noheader,nounits"], timeout=8)
        if rc == 0:
            ev["gpus"] = jobs.parse_probe("@@GPU\n" + out)["gpus"]
    ev["verdict"], ev["reasons"], ev["warnings"] = jobs.classify(ev)
    spy = find_py_spy() if pyspy else None
    ev["pyspy"] = spy
    if spy:
        pys = sorted([p for p in procs if "python" in (p.get("args") or "") + (p.get("exe") or "")],
                     key=lambda p: -(p.get("cpu") or 0))
        if pys:
            rc, out, err = jobs.run([spy, "dump", "--pid", str(pys[0]["pid"])], timeout=20)
            ev["pyspy_out"] = (out or err).splitlines()[:60]
    return ev


def stall_block(name, kind, silent_s, ev):
    lines = ["", "=" * 100,
             f"[ops guard] STALL{' ' + name if name else ''}: {kind} for {int(silent_s)} s",
             f"[ops guard] VERDICT: {ev.get('verdict')}"]
    for r in ev.get("reasons") or []:
        lines.append(f"[ops guard]   - {r}")
    for w in ev.get("warnings") or []:
        lines.append(f"[ops guard]   ! {w}")
    for p in ev.get("procs") or []:
        cpu = f"{p['cpu']:.0f}%" if p.get("cpu") is not None else "?"
        thr = ", ".join(f"{k}x{v}" for k, v in sorted((p.get("threads") or {}).items(), key=lambda kv: -kv[1])[:3])
        lines.append(f"[ops guard]   pid {p['pid']} {p.get('stat')} wchan={p.get('wchan') or '-'} cpu={cpu} "
                     f"threads[{thr}] {(p.get('args') or '')[:90]}")
        for f in p.get("nfs_fds") or []:
            lines.append(f"[ops guard]       open on NFS: {f}")
    if ev.get("gpus"):
        lines.append("[ops guard]   gpus: " + " ".join(f"#{g['index']}={g['util']:.0f}%" for g in ev["gpus"]))
    if ev.get("pyspy_out"):
        lines.append(f"[ops guard]   py-spy dump ({ev['pyspy']}):")
        lines += [f"[ops guard]     {l}" for l in ev["pyspy_out"]]
    elif not ev.get("pyspy"):
        lines.append("[ops guard]   (py-spy not installed; PYTHONFAULTHANDLER=1 is set: `kill -SIGABRT <pid>` "
                     "dumps Python stacks but kills the process)")
    lines.append("=" * 100)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------
# main loop


def kill_tree(proc, grace=10):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    t0 = time.time()
    while time.time() - t0 < grace and proc.poll() is None:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass


def guard(cmd, first_output=120.0, stall=300.0, log=None, kill=False, name=None, sample=2.0, out=None,
          poll=1.0, pyspy=True):
    """Run ``cmd`` under the stall watch. Returns the exit code (124 if killed for stalling)."""
    out = out or sys.stdout.buffer
    for w in start_warnings(cmd, log):
        out.write(f"[ops guard] WARNING: {w}\n".encode())
        out.flush()
    logf = open(log, "ab") if log else None
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True, bufsize=0)
    except FileNotFoundError:
        out.write(f"[ops guard] command not found: {cmd[0]}\n".encode())
        return 127
    lock = threading.Lock()
    state = {"last": None, "start": time.time()}

    def emit(data):
        with lock:
            out.write(data)
            out.flush()
            if logf:
                logf.write(data)
                logf.flush()

    def reader():
        while True:
            try:
                chunk = os.read(proc.stdout.fileno(), 65536)
            except OSError:
                break
            if not chunk:
                break
            state["last"] = time.time()
            emit(chunk)

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()
    next_report = {"first": first_output, "stall": stall}
    killed = False
    forwarded = {"n": 0}

    def forward(sig, _frame):
        forwarded["n"] += 1
        try:
            os.killpg(proc.pid, sig if forwarded["n"] < 3 else signal.SIGKILL)
        except OSError:
            pass

    old = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        while proc.poll() is None:
            time.sleep(poll)
            now = time.time()
            if state["last"] is None:
                silent, kind, key = now - state["start"], "no output since start", "first"
            else:
                silent, kind, key = now - state["last"], "no new output", "stall"
            if silent < next_report[key]:
                continue
            ev = diagnose(proc.pid, silent, sample=sample, pyspy=pyspy,
                          silent_threshold=first_output if key == "first" else stall)
            emit(stall_block(name, kind, silent, ev).encode())
            if kill:
                emit(f"[ops guard] --kill: terminating process group {proc.pid}\n".encode())
                kill_tree(proc)
                killed = True
                break
            next_report[key] = silent + max(next_report[key], 1.0)  # back off: report again after 2x
            if key == "first":
                next_report["stall"] = max(next_report["stall"], stall)
        rc = proc.wait()
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        rt.join(2)
        try:
            proc.stdout.close()
        except OSError:
            pass
        if logf:
            logf.close()
    if killed:
        return 124
    return rc if rc >= 0 else 128 - rc


def cmd_guard(args):
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("usage: ops guard [opts] -- CMD ...", file=sys.stderr)
        return 2
    return guard(cmd, args.first_output, args.stall, args.log, args.kill, args.name, pyspy=not args.no_py_spy)


def add_parser(sub):
    p = sub.add_parser("guard", help="run a long command with a no-output stall watch")
    p.add_argument("--first-output", type=float, default=120.0, help="s allowed before the first output")
    p.add_argument("--stall", type=float, default=300.0, help="s allowed without new output")
    p.add_argument("--log", help="also append the child's output here")
    p.add_argument("--kill", action="store_true", help="kill the process group on stall and exit 124")
    p.add_argument("--name", help="label for the STALL banner")
    p.add_argument("--no-py-spy", action="store_true")
    p.add_argument("cmd", nargs="...")
    p.set_defaults(func=cmd_guard)
