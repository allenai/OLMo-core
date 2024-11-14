from dataclasses import dataclass, field

import torch

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
            if (lr_field := self.scheduler.lr_field) not in group and (
                initial_lr_field := self.scheduler.initial_lr_field
            ) not in group:
                raise RuntimeError(
                    f"learning rate field '{lr_field}' and initial learning rate field "
                    f"'{initial_lr_field}' not found in optimizer param group"
                )

            # Ensure 'initial_lr' is set.
            if group.get(self.scheduler.initial_lr_field) is None:
                group[self.scheduler.initial_lr_field] = group["lr"]

            # Set new LR.
            new_lr = self.scheduler.get_lr(
                group[self.scheduler.initial_lr_field], self.step, self.trainer.max_steps
            )

            if isinstance(current_lr := group.get(self.scheduler.lr_field), torch.Tensor):
                current_lr.fill_(new_lr)
            else:
                group[self.scheduler.lr_field] = new_lr

            self.trainer.record_metric(
                f"optim/LR (group {group_idx})", group[self.scheduler.lr_field]
            )
