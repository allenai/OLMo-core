"""Probe whether the (shadowed) transformers can load Qwen3.5, and run a forward.

Used to find a transformers version that ships the `qwen3_5` architecture so Qwen3.5 can
plug into the HF/eval stack. Run under jobs/probe_transformers_qwen35.sh which installs a
candidate transformers into an isolated target and shadows it via PYTHONPATH.
"""
import glob
import sys
import traceback


def main():
    import torch
    import transformers
    print("transformers:", transformers.__version__, transformers.__file__, flush=True)
    for cls in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM",
                "Qwen3_5TextConfig", "Qwen3_5Config", "Qwen3NextForCausalLM"):
        print(f"  has {cls}: {hasattr(transformers, cls)}", flush=True)

    snap = sorted(glob.glob(
        "/scratch/users/prasann/huggingface-cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*/"))[-1]

    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(snap, trust_remote_code=True)
        print(f"AutoConfig OK: model_type={cfg.model_type}", flush=True)
    except Exception as e:
        print(f"AutoConfig FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)

    # try to actually load + forward (text only)
    loaded = None
    for name in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "AutoModelForCausalLM"):
        loader = getattr(transformers, name, None)
        if loader is None:
            continue
        try:
            print(f"loading via {name} ...", flush=True)
            m = loader.from_pretrained(snap, dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
            tok = transformers.AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
            ids = tok.encode("Hello! I am a test prompt.", return_tensors="pt").cuda()
            with torch.no_grad():
                out = m(input_ids=ids)
            lg = out.logits
            print(f"  LOADED+FORWARD via {name}: logits {tuple(lg.shape)} finite={torch.isfinite(lg).all().item()}",
                  flush=True)
            loaded = name
            break
        except Exception as e:
            print(f"  {name} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
    print(f"\nRESULT: {'LOADS via ' + loaded if loaded else 'CANNOT LOAD'}", flush=True)
    sys.exit(0 if loaded else 2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
