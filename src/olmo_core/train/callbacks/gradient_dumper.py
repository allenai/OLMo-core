import logging
from dataclasses import dataclass
from typing import Optional

from safetensors.torch import save_file

from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.distributed.utils import get_full_tensor, get_rank, is_distributed
from olmo_core.io import copy_dir, join_path
from olmo_core.utils import gc_cuda

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GradientDumperCallback(Callback):
    enabled: bool = True
    start_step: int = 0
    end_step: Optional[int] = None
    step_interval: int = 1
    save_first_n: Optional[int] = None

    def pre_optim_step(self):
        if not self.enabled:
            return

        if self.step < self.start_step:
            return

        if self.end_step is not None and self.step > self.end_step:
            return

        if (self.step - self.start_step) % self.step_interval != 0:
            return

        # Validate save_first_n
        if self.save_first_n is not None and self.save_first_n <= 0:
            raise ValueError(f"save_first_n must be positive, got {self.save_first_n}")

        output_dir = self.trainer.work_dir / "gradient_dumper"
        output_dir.mkdir(exist_ok=True, parents=True)

        step_dir = output_dir / f"step{self.step}"
        step_dir.mkdir(exist_ok=True, parents=True)

        assert hasattr(self.trainer.train_module, "model")
        model = getattr(self.trainer.train_module, "model")

        if self.save_first_n is None:
            # Save full gradients using distributed checkpoint
            full_grads_dir = step_dir / "full_gradients"

            grad_dict = {}
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_dict[name] = p.grad.detach()

            log.info(f"Saving {len(grad_dict)} gradient tensors for step {self.step}...")
            save_state_dict(
                full_grads_dir,
                grad_dict,
                save_overwrite=True,
            )
            log.info(f"Saved full gradients for step {self.step} to '{full_grads_dir}'")
        else:
            # Save first N elements of full gradients, gather all gradients and save from rank 0
            sampled_gradients_dir = step_dir / "sampled_gradients"
            sampled_gradients_dir.mkdir(exist_ok=True, parents=True)

            for name, p in model.named_parameters():
                if p.grad is None:
                    continue

                # get_full_tensor handles DTensor all-gather automatically
                full_grad = get_full_tensor(p.grad.detach())

                # Only rank 0 saves
                if get_rank() == 0:
                    full_grad = full_grad.cpu()

                    # Slice first N elements along dimension 0
                    dim_size = full_grad.shape[0]
                    actual_n = min(self.save_first_n, dim_size)
                    if actual_n < self.save_first_n:
                        log.warning(
                            f"Parameter '{name}': save_first_n={self.save_first_n} exceeds "
                            f"dimension size {dim_size}, capping to {actual_n}"
                        )

                    sliced_grad = full_grad.narrow(0, 0, actual_n)
                    slice_filename = f"{name}_first{actual_n}.safetensors"
                    slice_filepath = sampled_gradients_dir / slice_filename
                    save_file({"gradient": sliced_grad}, str(slice_filepath))
                    log.info(f"Saved first {actual_n} of '{name}' to '{slice_filepath}'")

                del full_grad

            log.info(f"Saved sampled gradients for step {self.step} to '{sampled_gradients_dir}'")

        # Persist directory to save_folder if different from work_dir
        # In distributed mode, only rank 0 uploads to avoid conflicts
        rel_step_dir = step_dir.relative_to(self.trainer.work_dir)
        target_dir = join_path(self.trainer.save_folder, rel_step_dir)
        if str(step_dir) != str(target_dir):
            if not is_distributed() or get_rank() == 0:
                log.info(f"Uploading gradients for step {self.step} to '{target_dir}'...")
                copy_dir(
                    step_dir, target_dir, save_overwrite=self.trainer.save_overwrite, quiet=True
                )
                log.info(f"Gradients for step {self.step} uploaded to '{target_dir}'")

        gc_cuda()
