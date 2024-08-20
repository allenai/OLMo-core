from dataclasses import dataclass

from olmo_core.optim.scheduler import ConstantScheduler, Scheduler

from .callback import Callback


@dataclass
class SchedulerCallback(Callback):
    scheduler: Scheduler = ConstantScheduler()

    def pre_optim_step(self):
        for group in self.trainer.optim.param_groups:
            if group.get("initial_lr") is None:
                group["initial_lr"] = group["lr"]
            group["lr"] = self.scheduler.get_lr(
                group["initial_lr"], self.step, self.trainer.max_steps
            )
