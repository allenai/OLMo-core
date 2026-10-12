"""Bisect the 32k-forward OOM in fit_summarizer_and_fidelity.py: hooks vs logits_to_keep vs base."""

import torch


def log(m):
    print(f"[bisect] {m}", flush=True)


def mem():
    return f"alloc {torch.cuda.memory_allocated()/2**30:.1f}GiB peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB"


def run(model, x, tag, hooks=False, ltk=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    handles = []
    stash = {}
    if hooks:
        handles = [
            model.model.layers[li].self_attn.register_forward_pre_hook(
                lambda m, a, kw, li=li: stash.__setitem__(
                    li, (a[0] if a else kw["hidden_states"]).detach()
                ),
                with_kwargs=True,
            )
            for li in range(0, 36, 3)
        ]
    try:
        with torch.no_grad():
            kwargs = {"use_cache": True}
            if ltk:
                kwargs["logits_to_keep"] = 1
            out = model(x, **kwargs)
        log(f"{tag}: OK  {mem()}")
        del out
    except torch.OutOfMemoryError as e:
        log(f"{tag}: OOM ({mem()})")
    finally:
        for h in handles:
            h.remove()
        stash.clear()
        torch.cuda.empty_cache()


def main():
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    log(f"model loaded {mem()}")
    x = torch.randint(1000, 50000, (1, 32768), device="cuda")
    run(model, x, "A base forward")
    run(model, x, "B + logits_to_keep=1", ltk=True)
    run(model, x, "C + 13 pre-hooks", hooks=True)
    run(model, x, "D + hooks + ltk", hooks=True, ltk=True)


if __name__ == "__main__":
    main()
