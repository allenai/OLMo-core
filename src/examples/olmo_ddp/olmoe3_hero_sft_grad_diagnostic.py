"""One real SFT backward, no optimizer update/checkpoint; isolate large gradients."""

import os

import olmoe3_hero_sft as sft
import olmoe3_hero_sft_plan as plan
import torch
import torch.distributed as dist
from olmoe3_hero_decay_runtime import verify_runtime
from olmoe3_lr_sweep_watch import log

from olmo_core.distributed.utils import get_rank
from olmo_core.train.trainer import Trainer


def install_gradient_hooks():
    from olmo_core.nn.attention import kda

    original = kda.dispatch_chunk_kda
    count = 0

    def report(name, value):
        value = value.detach().float()
        log(
            "SFT_BACKWARD_TENSOR",
            rank=get_rank(),
            name=name,
            norm=float(value.norm()),
            maximum=float(value.abs().max()),
            nonfinite=int((~torch.isfinite(value)).sum()),
        )

    def dispatch(**kwargs):
        nonlocal count
        number = count
        count += 1
        if torch.is_grad_enabled():
            for key in ("q", "k", "v", "g", "beta"):
                if kwargs[key].requires_grad:
                    kwargs[key].register_hook(lambda grad, n=f"kda{number}/{key}": report(n, grad))
        result = original(**kwargs)
        if torch.is_grad_enabled() and result[0].requires_grad:
            result[0].register_hook(lambda grad: report(f"kda{number}/output", grad))
        return result

    kda.dispatch_chunk_kda = dispatch


class GradientAudit(sft.SFTAudit):
    def pre_train(self):
        assert self.step == 0
        self.first_batch = True
        install_gradient_hooks()

    def pre_optim_step(self):
        assert self.step == 1
        optim = self.trainer.train_module.optim
        # Only copy/read the reduced grads: never clip or step the optimizer.
        optim._copy_model_grads_to_main_grads()
        rows = [
            {
                "name": name,
                "group": group,
                "placements": placements,
                "norm": float(value.float().norm()),
                "nonfinite": int((~torch.isfinite(value)).sum()),
            }
            for name, group, placements, value in optim._iter_local_grads()
        ]
        log(
            "SFT_GRADIENT_DIAGNOSTIC",
            rank=get_rank(),
            step=self.step,
            source="fresh-LC",
            top=sorted(rows, key=lambda row: row["norm"], reverse=True)[:25],
        )
        dist.barrier()
        raise SystemExit(0)


if __name__ == "__main__":
    verify_runtime()
    assert plan.MOUNT.is_mount()
    # Isolate every artifact from the sweep and its live restart smokes.
    plan.ROOT = plan.ROOT / "diagnostics" / os.environ["BEAKER_EXPERIMENT_ID"]
    sft.ROOT_WORK = plan.ROOT / "work"
    builder = sft.trainer_config

    def trainer_config(common):
        config = builder(common)
        config.callbacks.pop("sft_validation")
        config.callbacks["sft_audit"] = GradientAudit(run_id=common.run_name)
        config.callbacks["wandb"].enabled = False
        config.callbacks.pop("checkpoint_uploader", None)
        return config

    sft.trainer_config = trainer_config
    # This diagnostic has exactly one real batch; a mock backward adds no evidence.
    Trainer._dry_run_batch = lambda self: None
    sft.hero.qualified.apply_policy()
    sft.main(config_builder=sft.config_builder())
