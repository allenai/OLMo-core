from dataclasses import dataclass, field

from olmo_core.optim.scheduler import ConstantScheduler, Scheduler

from .callback import Callback


@dataclass
class SchedulerCallback(Callback):
    """
    Introduces a learning rate :class:`~olmo_core.optim.Scheduler` to the training loop.
    """

    scheduler: Scheduler = field(default_factory=ConstantScheduler)

    def pre_optim_step(self):
        for group_idx, group in enumerate(self.trainer.optim.param_groups):
            if group.get("initial_lr") is None:
                group["initial_lr"] = group["lr"]
            group["lr"] = self.scheduler.get_lr(
                group["initial_lr"], self.step, self.trainer.max_steps
            )
            self.trainer.record_metric(f"optim/LR (group {group_idx})", group["lr"])
