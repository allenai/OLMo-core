"""Token composition per length band for the built training shards (not just instance counts)."""
import glob, os, numpy as np
EOS = 248044
BANDS = [(0,2048),(2048,4096),(4096,8192),(8192,16384),(16384,32768),(32768,65536),(65536,131072),(131072,262145)]
LBL = ["<2k","2-4k","4-8k","8-16k","16-32k","32-64k","64-128k","128-256k"]
root = "/data/prasann/xlong5/shards_chunked"
print("task".ljust(14) + "".join(l.rjust(11) for l in LBL) + "      total")
grand = np.zeros(len(BANDS))
for d in sorted(glob.glob(root + "/*_train")):
    task = os.path.basename(d).replace("_train","")
    lens = []
    for f in sorted(glob.glob(d + "/token_ids_part_*.npy")):
        a = np.fromfile(f, dtype=np.uint32)
        idx = np.flatnonzero(a == EOS); prev = 0
        for i in idx:
            lens.append(int(i)-prev+1); prev = int(i)+1
    lens = np.array(lens)
    tok = np.array([lens[(lens>=lo)&(lens<hi)].sum() for lo,hi in BANDS], dtype=float)
    grand += tok
    print(task.ljust(14) + "".join(f"{t/1e6:>10.1f}M" for t in tok) + f"{lens.sum()/1e6:>10.1f}M")
print("-"*14 + "-"*(11*len(BANDS)) + "-"*11)
print("TOTAL".ljust(14) + "".join(f"{t/1e6:>10.1f}M" for t in grand) + f"{grand.sum()/1e6:>10.1f}M")
print("share".ljust(14) + "".join(f"{100*t/grand.sum():>10.1f}%" for t in grand))
