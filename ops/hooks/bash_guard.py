#!/usr/bin/env python3
"""Claude Code PreToolUse hook for the Bash tool (stdlib only, fails OPEN).

Reads the hook JSON on stdin. Denies (exit 2, reason on stderr) a small set of command shapes that
have repeatedly cost hours in this environment, and attaches non-blocking advice
(``hookSpecificOutput.additionalContext``) to job polling and job launches.

DENY:
  (i)   pytest / ``-m pytest`` / ``unittest discover`` / ``make checks|type-check`` on the login node
        (hostname ``radagast``) outside srun/sbatch -- except ``ops/tests``.
  (ii)  torch/vLLM launches (torchrun, ``-m torch.distributed.run``, vllm, anything inside srun/sbatch)
        whose interpreter is an NFS python (``/scratch/users/prasann/conda/...`` or ``/accounts/...``).
  (iii) ``/net/<node>/`` paths inside srun/sbatch/torchrun commands (job I/O over NFS).
  (iv)  bare ``torchrun`` (resolves to the NFS system miniforge).
  (v)   foreground polling: ``while``/``until`` loops containing ``sleep``, ``for`` loops sleeping >= 30 s,
        and ``sleep N`` (N >= 30) as a foreground command.

The parser is deliberately conservative: heredoc bodies are dropped, quoted strings are only
descended into for ``bash -c`` / ``sh -c`` / ``sbatch --wrap``, and any parse failure allows the command.
"""
import json
import os
import re
import shlex
import socket
import sys

LOGIN_HOSTS = {"radagast"}
OPS = "/accounts/projects/berkeleynlp/prasann/bin/ops"

NFS_PY = re.compile(
    r"(?:/scratch/users/prasann/conda/[^\s'\";]*|/accounts/[^\s'\";]*|/usr/local/linux/miniforge[^\s'\";]*"
    r"|/system/linux/miniforge[^\s'\";]*)/bin/(?:python[0-9.]*|torchrun|vllm)(?![\w.-])"
)
NET_PATH = re.compile(r"(?<![\w.-])/net/[a-z][\w-]*/")
# PATH pointed at a node-local env earlier in the same command makes a bare `torchrun` resolve there
NODE_LOCAL_PATH = re.compile(r"PATH=[\"']?(/data/|\$\{?\w*(ENV|PREFIX)\w*\}?/bin)")
REDIRECT = re.compile(r"^[0-9]*[<>&]*[<>][<>&]*$")
SEP_CHARS = ";&|\n()<>"

WRAPPER_FLAGS_WITH_VALUE = {
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "stdbuf": {"-i", "-o", "-e"},
    "env": {"-u", "--unset", "-C", "--chdir"},
}
KEYWORDS = {"do", "then", "else", "elif", "if", "!", "{", "}", "time", "nohup", "exec", "command",
            "builtin", "done", "fi"}

SRUN_BOOL = {"--overlap", "--pty", "--exclusive", "--parsable", "--wait", "-u", "--unbuffered", "-l",
             "--label", "-K", "-v", "-Q", "--quiet", "-X", "--test-only", "--hold", "-H", "-W",
             "--no-requeue", "--requeue", "--overcommit", "-O", "--exact", "--spread-job", "-Z",
             "--no-allocate", "--use-min-nodes", "--x11", "--mpi=none"}


# ----------------------------------------------------------------------------------------------
# tokenizing


def strip_heredocs(cmd):
    """Drop heredoc bodies (``<<EOF ... EOF``): their content is data/code, never shell commands."""
    out, lines, i = [], cmd.split("\n"), 0
    pat = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][\w]*)\1")
    while i < len(lines):
        line = lines[i]
        delims = [m.group(2) for m in pat.finditer(line)]
        out.append(pat.sub(" ", line))
        i += 1
        for d in delims:
            while i < len(lines) and lines[i].strip() != d:
                i += 1
            i += 1  # skip the delimiter line itself
    return "\n".join(out)


def strip_comments(cmd):
    return "\n".join(ln for ln in cmd.split("\n") if not ln.lstrip().startswith("#"))


def segments(cmd):
    """Split a shell string into simple commands (lists of words). Raises ValueError on bad quoting."""
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=SEP_CHARS)
    lex.whitespace = " \t\r"
    lex.commenters = ""
    lex.whitespace_split = True
    segs, cur, skip_next = [], [], False
    for tok in lex:
        if tok and all(c in SEP_CHARS for c in tok):
            if REDIRECT.match(tok) or (("<" in tok or ">" in tok) and not any(c in tok for c in ";|\n()")):
                skip_next = True  # redirect target is not a command word
                continue
            if cur:
                segs.append(cur)
            cur = []
            continue
        if skip_next:
            skip_next = False
            continue
        cur.append(tok)
    if cur:
        segs.append(cur)
    return segs


def strip_prefix(words):
    """Remove keywords, VAR=val assignments and transparent wrappers (timeout, env, nice, uv run...)."""
    w = list(words)
    changed = True
    while w and changed:
        changed = False
        head = w[0]
        if head in KEYWORDS or re.match(r"^[A-Za-z_]\w*=", head):
            w = w[1:]
            changed = True
        elif head in ("export", "local", "declare"):
            return []
        elif head in WRAPPER_FLAGS_WITH_VALUE:
            takes = WRAPPER_FLAGS_WITH_VALUE[head]
            w = w[1:]
            while w and (w[0].startswith("-") or re.match(r"^[A-Za-z_]\w*=", w[0])):
                flag = w.pop(0)
                if flag in takes and w:
                    w.pop(0)
            if head == "timeout" and w:
                w = w[1:]  # duration
            changed = True
        elif head == "uv" and len(w) > 1 and w[1] == "run":
            w = w[2:]
            changed = True
        elif head == "conda" and len(w) > 1 and w[1] == "run":
            w = w[2:]
            while w and w[0].startswith("-"):
                flag = w.pop(0)
                if flag in ("-n", "-p", "--name", "--prefix") and w:
                    w.pop(0)
            changed = True
    return w


def split_launcher(words):
    """For ``srun``/``sbatch`` words: return (options, payload)."""
    opts, i = [], 1
    while i < len(words):
        t = words[i]
        if not t.startswith("-") or t == "--":
            if t == "--":
                i += 1
            break
        opts.append(t)
        if "=" in t or t in SRUN_BOOL:
            i += 1
        elif t.startswith("--"):
            if i + 1 < len(words):
                opts.append(words[i + 1])
            i += 2
        elif len(t) == 2:
            if i + 1 < len(words):
                opts.append(words[i + 1])
            i += 2
        else:  # -wnode style
            i += 1
    return opts, words[i:]


def is_python(word):
    base = word.rsplit("/", 1)[-1]
    return bool(re.match(r"^python[0-9.]*$", base)) or bool(re.match(r"^\$\{?\w*PY\w*\}?$", word))


def parse_sleep_seconds(arg):
    m = re.match(r"^([0-9]*\.?[0-9]+)([smhd]?)$", arg or "")
    if not m:
        return None
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


# ----------------------------------------------------------------------------------------------
# analysis


class Findings:
    def __init__(self):
        self.deny = []
        self.advise = []

    def add_deny(self, msg):
        if msg not in self.deny:
            self.deny.append(msg)

    def add_advise(self, msg):
        if msg not in self.advise:
            self.advise.append(msg)


MSG = {
    "tests": "DENIED: test suites/type checks are CPU-heavy and must not run on the login node "
             "(radagast). Run them on a compute node: `srun --partition=berkeleynlp "
             "--qos=preemptive_high_sewonm --nodelist=horton --cpus-per-task=16 --mem=64G --time=00:30:00 "
             "/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python -m pytest ...`.",
    "nfs_py": "DENIED: this job launch uses an NFS interpreter ({path}); imports page GBs over ~5 MB/s "
              "NFS and the job parks in D state (nfs_wait_bit_killable) for 18+ min. Use the node-local "
              "env under /data/prasann/... (e.g. /data/prasann/conda/envs/<env>/bin/python or "
              "/data/prasann/ctc_vllm_venv/bin/python).",
    "net": "DENIED: job I/O through /net/<node>/ ({path}) is the slow NFS layer and deadlocks under "
           "load. Inside a job, use the node's own /data/... path; /net is for read-only inspection "
           "from the login node.",
    "torchrun": "DENIED: bare `torchrun` resolves to the NFS system miniforge "
                "(/usr/local/linux/miniforge-3.13) and stalls in D state. Use "
                "`$PY -m torch.distributed.run ...` with PY=/data/prasann/conda/envs/<env>/bin/python.",
    "loop": "DENIED: foreground polling loop (while/until/for + sleep). It blocks the session and leaves "
            "orphaned watchers. Take ONE snapshot with `" + OPS + " status` (cached, <15 s), use the "
            "Monitor tool, or run the wait with run_in_background=true (and give it an exit bound).",
    "sleep": "DENIED: foreground `sleep {n}` (>= 30 s). Don't wait in the foreground: take one snapshot "
             "with `" + OPS + " status`, use the Monitor tool, or use run_in_background=true.",
}

ADVICE_POLL = ("Polling job state? Prefer `" + OPS + " status` (one cached snapshot of slurm + Beaker + "
               "Lambda joined to the run registry, flags OVERRUN/STALE-LOG/PENDING>1h) or `" + OPS +
               " stall <job>` for a hung-looking job, instead of repeated squeue/sacct/beaker calls.")
ADVICE_NET_COPY = ("Note: copying/inspecting via /net/<node>/ inside a job runs at NFS speed (~5 MB/s) and can "
                   "wedge in nfs_wait_bit_killable; keep the job's real I/O on its own /data.")
# commands that may legitimately touch /net/<node>/ inside a job (one-off staging copies, inspection)
NET_COPY_OK = {"rsync", "cp", "ls", "df", "du", "stat", "head", "tail", "cat", "wc", "grep", "find", "test",
               "[", "echo", "mkdir", "readlink", "realpath", "md5sum", "sha256sum", "tar"}
ADVICE_LAUNCH = ("Launching a job: register it (`" + OPS + " register --run-id <run> --experiment <exp> "
                 "--cluster <slurm|beaker|lambda> --job-id <id> --expected-minutes N --log-path <log>`), "
                 "smoke-test first, log to node-local /data (never --output on /accounts or /scratch), "
                 "log progress at items 1,2,5,10, and consider wrapping the payload in `" + OPS +
                 " guard --stall 300 -- ...`.")


def _is_test_segment(w):
    if not w:
        return False
    head = w[0].rsplit("/", 1)[-1]
    if any(x in w for x in ("--version", "--help", "-h")):
        return False
    if head in ("pytest", "py.test"):
        return True
    if is_python(w[0]) and "-m" in w:
        i = w.index("-m")
        if i + 1 < len(w):
            mod = w[i + 1]
            if mod == "pytest":
                return True
            if mod == "unittest" and "discover" in w[i + 2:]:
                return True
    if head == "make" and any(t in ("checks", "type-check") for t in w[1:]):
        return True
    return False


def _is_torch_launch(w):
    if not w:
        return False
    head = w[0].rsplit("/", 1)[-1]
    if head in ("torchrun", "vllm"):
        return True
    if is_python(w[0]) and "-m" in w:
        i = w.index("-m")
        if i + 1 < len(w) and (w[i + 1] == "torch.distributed.run" or w[i + 1].startswith("vllm")):
            return True
    return False


def analyze(cmd, f, ctx, depth=0):
    """Analyze a shell string. ``ctx`` keys: job (inside srun/sbatch), login, background, cwd, raw."""
    if depth > 4:
        return
    text = strip_comments(strip_heredocs(cmd))
    try:
        segs = segments(text)
    except ValueError:
        return  # unbalanced quotes etc.: fail open
    loops = []  # stack of open while/until/for loops
    for raw in segs:
        if raw and raw[0] in ("while", "until", "for"):
            loops.append(raw[0])
            w = strip_prefix(raw[1:])
        else:
            if raw and raw[0] == "done" and loops:
                loops.pop()
            w = strip_prefix(raw)

        # inside a job (srun payload / sbatch --wrap / sbatch script): every word counts, including
        # assignments like PY=/scratch/.../bin/python that strip_prefix drops
        head = w[0].rsplit("/", 1)[-1] if w else ""
        if ctx["job"]:
            if head in ("bash", "sh", "zsh") and "-c" in raw:
                _scan_job_words(raw[: raw.index("-c")], f, ctx)  # the -c string is analyzed below
            elif head not in ("srun", "sbatch"):  # launcher options/payload are handled below
                _scan_job_words(raw, f, ctx, net_ok=head in NET_COPY_OK)
        if not w:
            continue

        # (v) polling
        if head == "sleep" and len(w) > 1 and not ctx["job"]:
            n = parse_sleep_seconds(w[1])
            if n is not None:
                looping = any(k in ("while", "until") for k in loops) or ("for" in loops and n >= 30)
                if looping:
                    (f.add_advise if ctx["background"] else f.add_deny)(MSG["loop"])
                elif n >= 30:
                    (f.add_advise if ctx["background"] else f.add_deny)(MSG["sleep"].format(n=w[1]))

        # nested shells
        if head in ("bash", "sh", "zsh") and "-c" in w:
            i = w.index("-c")
            if i + 1 < len(w):
                analyze(w[i + 1], f, ctx, depth + 1)
            continue

        # advice: polling
        if head in ("squeue", "sacct") or (
            head == "beaker" and len(w) > 2 and (w[1], w[2]) in {
                ("experiment", "get"), ("job", "list"), ("job", "logs"), ("job", "get"),
                ("experiment", "logs"), ("workspace", "experiments")}
        ):
            f.add_advise(ADVICE_POLL)

        # launches
        if head in ("srun", "sbatch"):
            opts, payload = split_launcher(w)
            inspect = head == "srun" and _is_inspection_srun(opts)
            jctx = dict(ctx, job=True, inspect=inspect or ctx.get("inspect", False))
            wrap = None
            plain_opts = []
            for j, o in enumerate(opts):
                if o.startswith("--wrap="):
                    wrap = o[len("--wrap="):]
                elif o == "--wrap" and j + 1 < len(opts):
                    wrap = opts[j + 1]
                elif j > 0 and opts[j - 1] == "--wrap":
                    continue
                else:
                    plain_opts.append(o)
            _scan_job_words(plain_opts, f, jctx)
            if head == "sbatch":
                f.add_advise(ADVICE_LAUNCH)
                if wrap is not None:
                    analyze(wrap, f, jctx, depth + 1)
                elif payload:
                    _scan_job_words(payload[1:], f, jctx)
                    scan_script(payload[0], f, ctx.get("cwd"))
            elif payload:
                analyze(shlex.join(payload), f, jctx, depth + 1)
            continue

        if head == "gantry" or (head == "beaker" and len(w) > 2 and w[2] in ("create", "launch")):
            f.add_advise(ADVICE_LAUNCH)
            continue
        if head in ("ssh", "scp", "rsync", "git", "echo", "printf", "grep", "rg", "cat", "sed", "awk",
                    "ls", "which", "type", "pgrep", "pkill", "ps", "tail", "head", "less", "stat"):
            continue
        if is_python(w[0]) and any(x == "launch" for x in w[2:4]):
            f.add_advise(ADVICE_LAUNCH)

        # (iv) bare torchrun
        if (w[0] == "torchrun" and not NODE_LOCAL_PATH.search(ctx["raw"])) or \
                re.match(r"^/(usr/local|system)/linux/miniforge[^/]*/bin/torchrun$", w[0]):
            f.add_deny(MSG["torchrun"])

        # (ii)/(iii) torch/vLLM launch outside a job: NFS interpreter or /net paths
        if _is_torch_launch(w) and not ctx["job"]:
            _scan_job_words(raw, f, ctx)

        # (i) tests on the login node
        if ctx["login"] and not ctx["job"] and _is_test_segment(w):
            raw_all = ctx["raw"]
            exempt = "ops/tests" in raw_all or re.search(r"\bcd\s+\S*\bops/?\s*(&&|;|\n)", raw_all)
            if not exempt:
                f.add_deny(MSG["tests"])


def _scan_job_words(words, f, ctx, net_ok=False):
    for t in words:
        m = NFS_PY.search(t)
        if m:
            f.add_deny(MSG["nfs_py"].format(path=m.group(0)))
        m = NET_PATH.search(t)
        if m and not ctx.get("inspect"):
            if net_ok:
                f.add_advise(ADVICE_NET_COPY)
            else:
                f.add_deny(MSG["net"].format(path=t[:80]))


def _is_inspection_srun(opts):
    """``srun --jobid=X --overlap ...`` attaches to a running job for diagnosis (allowed to read /net)."""
    return any(o.startswith("--jobid") for o in opts) and "--overlap" in opts


def scan_script(path, f, cwd):
    """Advisory scan of an sbatch script file: never denies (file-based parsing is less reliable)."""
    try:
        p = path if os.path.isabs(path) else os.path.join(cwd or os.getcwd(), path)
        if not os.path.isfile(p) or os.path.getsize(p) > 200_000:
            return
        with open(p, errors="replace") as fh:
            text = fh.read()
    except Exception:
        return
    notes = []
    for line in text.split("\n"):
        m = re.match(r"^#SBATCH\s+(?:--output|--error|-o|-e)[=\s]+(\S+)", line)
        if m and re.match(r"^/(accounts|scratch|net)/", m.group(1)):
            notes.append(f"slurm log {m.group(1)} is on NFS (known step-0 deadlock); use /data/prasann/joblogs/")
    sub = Findings()
    analyze(text, sub, {"job": True, "login": False, "background": False, "cwd": cwd, "raw": text}, 1)
    for d in sub.deny:
        notes.append(d.replace("DENIED: ", ""))
    if notes:
        f.add_advise(f"Script {path} looks risky (advisory, not blocked): " + " | ".join(notes[:4]))


def check(command, hostname=None, background=False, cwd=None):
    """Pure entry point: returns a Findings for ``command``."""
    f = Findings()
    host = (hostname or socket.gethostname()).split(".")[0]
    ctx = {"job": False, "login": host in LOGIN_HOSTS, "background": bool(background), "cwd": cwd,
           "raw": command}
    analyze(command, f, ctx)
    return f


def main():
    try:
        data = json.load(sys.stdin)
        if data.get("tool_name") != "Bash":
            return 0
        ti = data.get("tool_input") or {}
        cmd = ti.get("command") or ""
        f = check(cmd, os.environ.get("OPS_HOOK_HOSTNAME"), ti.get("run_in_background"), data.get("cwd"))
        if f.deny:
            sys.stderr.write("[ops bash_guard] " + "\n".join(f.deny) + "\n")
            return 2
        if f.advise:
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                                     "additionalContext": "\n".join(f.advise)}}))
        return 0
    except Exception:
        return 0  # fail open


if __name__ == "__main__":
    sys.exit(main())
