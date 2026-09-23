"""
Where does a landmark checkpoint's ~10 s per-example "setup" go?

The native eval's timing log (OLMO_GEN_LOG_TIMING=1) shows prefill+decode at ~0.5 s per example on
a fast-compressive-landmark Qwen3.5-4B, but ~10 s of "setup" before the first decode step -- flat
from 3k to 16k tokens, and occasionally ~0.1 s. For landmark models the real prefill runs inside
that window, so the 10 s is something shape-dependent that is sometimes cached. This builds the
generation module exactly as eval_lc_native.py does and times prefill over controlled prompt
lengths (exact repeats, +1 token, +16 tokens, new lengths), cProfiling the slow ones.

    TRITON_PRINT_AUTOTUNING=1 python debug/ctc_eval_speed/profile_landmark_setup.py --ckpt <step dir>
"""

import argparse
import cProfile
import io
import os
import pstats
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lengths", default="3000,3000,3001,3016,3100,8000,8000,8001,16000,16001")
    args = ap.parse_args()

    from olmo_core.config import DType
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import (
        TransformerGenerationModuleConfig,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    gen_cfg = GenerationConfig(eos_token_id=248046, pad_token_id=248044, max_length=270000,
                               use_cache=True)
    t0 = time.time()
    gm = TransformerGenerationModuleConfig(gen_cfg, float8_config=None, dtype=DType("bfloat16"),
                                           compile_model=False).build(checkpoint_dir=args.ckpt,
                                                                      device=device)
    print(f"built in {time.time() - t0:.1f}s", flush=True)
    g = torch.Generator().manual_seed(0)
    for i, L in enumerate(int(x) for x in args.lengths.split(",")):
        ids = torch.randint(1000, 200000, (1, L), generator=g).to(device)
        prof = cProfile.Profile()
        torch.cuda.synchronize()
        t = time.perf_counter()
        prof.enable()
        gm.generate_batch(input_ids=ids, attention_mask=torch.ones_like(ids), completions_only=False,
                          log_timing=False, max_new_tokens=4)
        torch.cuda.synchronize()
        prof.disable()
        dt = time.perf_counter() - t
        print(f"[{i}] len={L}  total {dt:.2f}s", flush=True)
        if dt > 3:
            s = io.StringIO()
            pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(30)
            print(s.getvalue()[:6000], flush=True)


if __name__ == "__main__":
    main()
