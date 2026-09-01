"""Audit every tokenized arm on weka before its training job reaches the head of the queue.

A shard that trips the trainer's guards (max_example_len above --seq-len, a task string that does
not match the launcher's --task, the Qwen3 eos instead of Qwen3.5's 248044) fails the run minutes
after it finally gets a GPU -- which, on a cluster where a job waits hours to place, is the most
expensive kind of failure. This is a two-second CPU check for all of it.
"""
import glob
import json
import os

ROOT = os.environ.get(
    "TASKSCALE_ARMS",
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/taskscale_lengthmix/arms_tokenized",
)
SEQ_LEN = 65536
EOS = 248044

bad = 0
for d in sorted(glob.glob(f"{ROOT}/*")):
    name = os.path.basename(d)
    meta = os.path.join(d, "metadata.json")
    if not os.path.exists(meta):
        print(f"{name:34s} NO METADATA")
        bad += 1
        continue
    j = json.load(open(meta))
    parts = len(glob.glob(os.path.join(d, "token_ids_part_*.npy")))
    masks = len(glob.glob(os.path.join(d, "labels_mask_*.npy")))
    flags = []
    # A shard written before max_example_len existed is the dangerous case, not a cosmetic one:
    # the trainer guards --seq-len with meta.get("max_example_len", 0), so a missing key means the
    # guard reads 0, never fires, and PadToLength silently DROPS every over-length example.
    mx = j.get("max_example_len", j.get("max_len"))
    if mx is None:
        flags.append("no max_example_len -- the trainer's seq-len guard cannot fire")
    elif mx > SEQ_LEN:
        flags.append(f"max_example_len {mx} > seq-len {SEQ_LEN}")
    if j["eos"] != EOS:
        flags.append(f"eos {j['eos']} != {EOS}")
    if j["num_skipped"]:
        flags.append(f"{j['num_skipped']} skipped")
    if parts != 1 or masks != 1:
        flags.append(f"{parts} token parts / {masks} mask parts")
    bad += bool(flags)
    print(
        f"{name:34s} task={j['task']:13s} inst={j['num_instances']:6d} "
        f"tok={j['num_tokens'] / 1e6:6.1f}M median={j['median_len']:6d} "
        f"max={mx if mx is not None else -1:6d}  {'; '.join(flags) if flags else 'OK'}"
    )
print(f"\n{bad} arm(s) flagged")
