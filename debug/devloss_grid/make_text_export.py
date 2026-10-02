"""Turn a vLLM *serving copy* of a Qwen3.5 export back into a text-only HF checkpoint that
``olmo_core.nn.hf.checkpoint.load_hf_model`` can read (model_type ``qwen3_5_text``).

The suite's dense 4B checkpoints survive locally only as serving copies
(``/data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-<task>-full`` on cubbins): a VL wrapper
``config.json`` with the real text config under ``text_config``, one F32 ``model.safetensors``
whose language-model weights are keyed ``model.language_model.*`` plus ~600 synthesized
``visual.*``/``model.visual.*`` tensors (CLAUDE.md "Loading an olmo-exported Qwen3.5 in vLLM").
This reverses that: keep ``model.language_model.*`` -> ``model.*`` and ``lm_head.weight``, drop
every visual tensor, cast to bf16, and write ``config.json`` = ``text_config``. Streams tensors with
``safe_open`` so the 17 GB F32 source is read once (fine over /net) and never held in RAM.

    python make_text_export.py --src /net/cubbins/.../ctc-4b-nq-full --dst /data/prasann/devloss_grid/ckpts/ctc-4b-nq-full
"""
import argparse, json, os, shutil, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
a = ap.parse_args()
if os.path.exists(f"{a.dst}/.text_export_done"):
    print(f"[text-export] {a.dst} already complete, skipping"); raise SystemExit(0)
os.makedirs(a.dst, exist_ok=True)
cfg = json.load(open(f"{a.src}/config.json"))
tc = cfg.get("text_config", cfg)
assert tc.get("model_type") == "qwen3_5_text", tc.get("model_type")
tc = dict(tc); tc["architectures"] = ["Qwen3_5ForCausalLM"]; tc["dtype"] = a.dtype
json.dump(tc, open(f"{a.dst}/config.json", "w"), indent=2)
for f in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
    if os.path.exists(f"{a.src}/{f}"):
        shutil.copy(f"{a.src}/{f}", f"{a.dst}/{f}")
if not os.path.exists(f"{a.dst}/generation_config.json"):  # load_hf_model asserts one of these exists
    json.dump({"eos_token_id": tc.get("eos_token_id"), "pad_token_id": tc.get("pad_token_id")},
              open(f"{a.dst}/generation_config.json", "w"))
dt = getattr(torch, a.dtype)
out, n_lm, n_drop, t0 = {}, 0, 0, time.time()
srcs = sorted(f for f in os.listdir(a.src) if f.endswith(".safetensors"))
for sf in srcs:
    with safe_open(f"{a.src}/{sf}", framework="pt", device="cpu") as fh:
        for k in fh.keys():
            if k.startswith("model.language_model."):
                out["model." + k[len("model.language_model."):]] = fh.get_tensor(k).to(dt).contiguous(); n_lm += 1
            elif k == "lm_head.weight":
                out[k] = fh.get_tensor(k).to(dt).contiguous(); n_lm += 1
            else:
                n_drop += 1
            if (n_lm + n_drop) % 100 == 0:
                print(f"[text-export] {n_lm} kept / {n_drop} dropped ({time.time()-t0:.0f}s)", flush=True)
assert n_lm >= 400, f"only {n_lm} language-model tensors found -- not a serving copy?"
save_file(out, f"{a.dst}/model.safetensors", metadata={"format": "pt"})
open(f"{a.dst}/.text_export_done", "w").write(json.dumps({"src": a.src, "kept": n_lm, "dropped": n_drop, "dtype": a.dtype}))
print(f"[text-export] wrote {a.dst}/model.safetensors: {n_lm} tensors kept, {n_drop} visual dropped, {time.time()-t0:.0f}s", flush=True)
