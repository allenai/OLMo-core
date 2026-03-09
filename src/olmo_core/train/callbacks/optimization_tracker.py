"""Callback to track optimization diagnostics (residuals, gradients, params, norms)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

log = logging.getLogger(__name__)

from olmo_core.nn.layer_norm import LayerNorm, RMSNorm
from olmo_core.nn.residual_stream import ResidualStream
from olmo_core.distributed.utils import get_full_tensor, get_local_tensor

from ..common import MetricMergeStrategy, ReduceType
from .callback import Callback


@dataclass
class OptimizationDiagnosticsCallback(Callback):
    """
    Tracks optimization-related diagnostics.

    IMPORTANT: Parameter metrics (norm weight scale/dist) use :func:`get_full_tensor()`
    and reflect the true global statistic. Activation metrics use :func:`get_local_tensor()`
    and log with reduce/merge = mean, so the final logged value is the mean of per-rank
    local-shard statistics, NOT the true global statistic.

    Ratios computed from local activation norms can differ from ratios computed on the full
    tensors, especially under tensor/model parallelism.
    """

    enabled: bool = False
    log_interval: Optional[int] = None
    eps: float = 1e-8  # for ratios and norms.

    track_residual_updates: bool = False  # Log residual norm, update norm, and ratio per ResidualStream.

    eps_hit_tolerance: float = 0.1  # Extra fractional slack above eps to count as a hit.
    track_layer_norm_eps: bool = False  # Log eps-hit indicators for LayerNorm/RMSNorm.
    track_norm_weight_scale: bool = False  # Log RMS and abs_max of effective norm weight per RMSNorm module.
    track_norm_weight_dist: bool = False  # Log histogram of effective norm weight per RMSNorm module to W&B.

    track_param_grad_rmse: bool = False  # Log per-parameter grad RMSE.
    track_param_grad_meanvar: bool = False  # Log per-parameter grad mean/stddev.

    track_param_meanvar: bool = False  # Log per-parameter mean and stddev.
    track_update_param_ratio: bool = False  # Log per-parameter update/param ratio.

    track_activation_rmse: bool = False  # Log activation RMSE over all modules.
    track_activation_meanvar: bool = False  # Log activation mean/stddev over all modules.
    track_activation_norm: bool = False  # Log mean activation norm over batch/tokens for all modules.

    track_activation_grad_rmse: bool = False  # Log activation grad RMSE over all modules.
    track_activation_grad_meanvar: bool = False  # Log activation grad mean/stddev over all modules.
    track_activation_grad_norm: bool = False  # Log mean activation grad norm over batch/tokens for all modules.

    track_update_rmse: bool = False  # Log per-parameter update RMSE.

    track_optimizer_state_rmse_meanvar: bool = False  # Log optimizer state RMSE/mean/stddev.

    track_lm_head: bool = False  # Log LM head metrics (top-1 mass, max/avg/min logits).

    track_param_movement: bool = False  # Log fraction of parameters moving beyond threshold.
    param_movement_threshold: float = 0.01  # Threshold for "significant" parameter movement (0.1% by default).
    
    track_embedding_usage: bool = False  # Log embedding usage statistics (num activated, avg grad, activation counts).
    
    track_gradient_outliers: bool = False  # Log when gradients exceed mu ± k*sigma (uses AdamW exp_avg/exp_avg_sq)
    gradient_outlier_k: List[float] = field(default_factory=lambda: [4.0, 6.0])  # k-sigma thresholds for outlier detection
    
    namespace: str = "optim_diagnostics"  # Metric namespace prefix.

    _handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list, repr=False)
    _prev_params: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _param_name_map: Optional[Dict[int, str]] = field(default=None, repr=False)

    _embedding_token_ids: List[torch.Tensor] = field(default_factory=list, repr=False)  # Token IDs seen in forward pass
    _embedding_vocab_size: Optional[int] = field(default=None, repr=False)

    def post_attach(self):
        if not self.enabled:
            return

        model = getattr(self.trainer.train_module, "model", None)
        if model is None or not isinstance(model, nn.Module):
            return

        self._param_name_map = {id(p): n for n, p in model.named_parameters()}

        if self.track_residual_updates:
            for name, module in model.named_modules():
                if isinstance(module, ResidualStream):
                    self._handles.append(
                        module.register_forward_hook(self._make_residual_stream_forward_hook(name))
                    )
                    self._handles.append(
                        module.register_full_backward_hook(
                            self._make_residual_stream_backward_hook(name)
                        )
                    )

        if (
            self.track_activation_rmse
            or self.track_activation_meanvar
            or self.track_activation_norm
            or self.track_activation_grad_rmse
            or self.track_activation_grad_meanvar
            or self.track_activation_grad_norm
        ):
            for name, module in model.named_modules():
                self._handles.append(
                    module.register_forward_hook(self._make_activation_forward_hook(name))
                )
                self._handles.append(
                    module.register_full_backward_hook(self._make_activation_backward_hook(name))
                )

        if self.track_layer_norm_eps:
            for name, module in model.named_modules():
                if isinstance(module, LayerNorm):
                    self._handles.append(
                        module.register_forward_hook(self._make_layer_norm_hook(name))
                    )

        if self.track_lm_head:
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and isinstance(lm_head, nn.Module):
                self._handles.append(
                    lm_head.register_forward_hook(self._make_lm_head_hook())
                )

        if self.track_embedding_usage:
            embeddings = getattr(model, "embeddings", None)
            if embeddings is not None and isinstance(embeddings, nn.Embedding):
                self._embedding_vocab_size = embeddings.num_embeddings
                self._handles.append(
                    embeddings.register_forward_hook(self._make_embedding_forward_hook())
                )

    def post_train(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._prev_params = None
        self._embedding_token_ids.clear()

    def pre_optim_step(self):
        if not self._should_log():
            return

        model = getattr(self.trainer.train_module, "model", None)
        if model is None or not isinstance(model, nn.Module):
            return

        optim = getattr(self.trainer.train_module, "optim", None)
        if self.track_gradient_outliers:
            self._check_gradient_outliers(model, optim)

        if self.track_param_grad_rmse or self.track_param_grad_meanvar:
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad = get_local_tensor(param.grad.detach()).float()
                if self.track_param_grad_rmse:
                    grad_rmse = self._rmse(grad)
                    self._log_metric(f"grads/{name}/rmse", grad_rmse)
                if self.track_param_grad_meanvar:
                    grad_mean, grad_std = self._meanstd(grad)
                    self._log_metric(f"grads/{name}/mean", grad_mean)
                    self._log_metric(f"grads/{name}/stddev", grad_std)

        if self.track_update_param_ratio or self.track_param_movement:
            self._prev_params = {
                name: get_local_tensor(p.detach()).float().clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }

    def _should_log(self) -> bool:
        if not self.enabled:
            return False
        interval = self.log_interval or self.trainer.metrics_collect_interval
        if interval <= 0:
            return False
        return self.step % interval == 0

    def _log_metric(
        self,
        path: str,
        value: torch.Tensor,
        *,
        reduce_type: ReduceType = ReduceType.mean,
        merge_strategy: MetricMergeStrategy = MetricMergeStrategy.mean,
    ) -> None:
        self.trainer.record_metric(
            f"{self.namespace}/{path}",
            value,
            reduce_type=reduce_type,
            merge_strategy=merge_strategy,
        )

    def _rmse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute RMSE over all elements in the tensor."""
        return tensor.pow(2).mean().sqrt()

    def _meanstd(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean/stddev over all elements in the tensor."""
        mean = tensor.mean()
        std = tensor.var(unbiased=False).sqrt()
        return mean, std

    def _check_gradient_outliers(self, model: nn.Module, optim):
        """Check for gradient outliers (values beyond mu ± k*sigma) per parameter using AdamW state."""
        # This must be called at pre_optim_step or later, where FSDP has completed
        # gradient all-reduce and the optimizer has not yet updated exp_avg / exp_avg_sq.
        # Reads of grad and optimizer state are therefore coherent without an explicit
        # stream sync. Moving this to an earlier hook may require adding one.
        if optim is None or not hasattr(optim, 'state'):
            return

        # Check if optimizer is AdamW-like (has exp_avg and exp_avg_sq)
        # Works with torch.optim.AdamW and custom AdamW implementations
        if not any('exp_avg' in state and 'exp_avg_sq' in state for state in optim.state.values()):
            return

        # Build param -> betas map so bias correction uses the correct per-group betas.
        param_to_betas: dict = {}
        missing_betas_warned: set = set()
        for group in optim.param_groups:
            if "betas" not in group:
                continue
            for p in group["params"]:
                param_to_betas[id(p)] = group["betas"]

        for param in optim.state.keys():
            name = None
            if self._param_name_map is not None:
                name = self._param_name_map.get(id(param))
            if name is None:
                name = f"param_{id(param)}"

            if param.grad is None:
                continue

            state = optim.state[param]
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            step_tensor = state.get("step")

            # Need both moments to compute mean and std
            if exp_avg is None or exp_avg_sq is None:
                continue

            grad = get_local_tensor(param.grad.detach()).float()
            mean_est = get_local_tensor(exp_avg.detach()).float()
            second_moment = get_local_tensor(exp_avg_sq.detach()).float()

            # Apply bias correction so the estimates are accurate regardless of beta2.
            # The stored exp_avg / exp_avg_sq are raw EMAs initialised at zero, so they
            # underestimate the true moments by (1 - beta^t) until fully warmed up.
            # This matters especially with large beta2 values (e.g. 0.999).
            if step_tensor is not None:
                betas = param_to_betas.get(id(param))
                if betas is None:
                    if name not in missing_betas_warned:
                        log.warning(
                            "Could not find betas for param '%s' in optimizer param groups; "
                            "skipping bias correction for gradient outlier detection.",
                            name,
                        )
                        missing_betas_warned.add(name)
                else:
                    beta1, beta2 = betas
                    step = step_tensor.item()
                    if step > 0:
                        mean_est = mean_est / (1 - beta1 ** step)
                        second_moment = second_moment / (1 - beta2 ** step)

            # Variance = E[X^2] - E[X]^2
            variance = second_moment - mean_est.pow(2)
            variance = torch.clamp(variance, min=0)  # Ensure non-negative due to numerical issues
            std_est = variance.sqrt()

            for k in self.gradient_outlier_k:
                lower_bound = mean_est - k * std_est
                upper_bound = mean_est + k * std_est

                # Count outliers - compute fraction directly to avoid sync
                outlier_mask = (grad < lower_bound) | (grad > upper_bound)
                outlier_frac = outlier_mask.float().mean()

                # Log fraction (will be 0.0 if no outliers)
                self._log_metric(
                    f"gradient_outliers/{name}/{int(k)}sigma_fraction",
                    outlier_frac,
                )

    def _mean_norm_over_batch_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute mean L2 norm over batch/tokens (all dims except the last)."""
        return tensor.norm(dim=-1).mean()

    def _vector_rmse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute RMSE per vector (last dim), then average over batch/tokens."""
        return tensor.pow(2).mean(dim=-1).sqrt().mean()

    def _vector_meanstd(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact per-vector mean and stddev (over last dim), averaged over batch/tokens.

        Expects a fully-gathered tensor (not a TP shard). Callers should use
        ``get_full_tensor()`` before passing when the tensor may be sharded on
        the last dim under tensor parallelism.
        """
        mean = tensor.mean(dim=-1).mean()
        std = tensor.var(dim=-1, unbiased=False).sqrt().mean()
        return mean, std

    def _log_embedding_metrics(self):
        # Always log metrics on all ranks to avoid collective mismatch.
        # Under data parallelism each rank sees a different micro-batch, so each
        # rank only counts activations from its own portion of the global batch.
        # We accumulate per-rank counts into a vocab-sized tensor and all_reduce
        # (SUM) so the resulting counts reflect the full global batch.
        device = self.trainer.device
        vocab_size = self._embedding_vocab_size

        if vocab_size is None:
            return

        # Build local per-token activation counts as a [vocab_size] tensor.
        # Concatenate all batches first to issue a single bincount kernel rather
        # than one scatter_add_ per gradient-accumulation microstep.
        if self._embedding_token_ids:
            all_ids = torch.cat([t.view(-1) for t in self._embedding_token_ids]).to(
                device=device, dtype=torch.long
            )
            counts = torch.bincount(all_ids, minlength=vocab_size).float()
        else:
            counts = torch.zeros(vocab_size, dtype=torch.float32, device=device)

        # Reduce across all ranks to get global counts.
        # After this, all ranks hold identical global counts.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        # num_activated: number of token IDs with at least one activation globally.
        num_activated = (counts > 0).sum().float()
        self.trainer.record_metric(
            f"{self.namespace}/embeddings/num_activated",
            num_activated,
            reduce_type=None,  # already globally reduced above
        )

        # Median activation count over the activated tokens.
        activated_counts = counts[counts > 0]
        if activated_counts.numel() > 0:
            median_count = torch.median(activated_counts)
        else:
            median_count = torch.tensor(0.0, device=device)
        self.trainer.record_metric(
            f"{self.namespace}/embeddings/median_activation_count",
            median_count,
            reduce_type=None,  # already globally reduced above
        )

    def _make_residual_stream_forward_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, torch.Tensor], output: torch.Tensor):
            del module
            if not self._should_log():
                return
            if not isinstance(output, torch.Tensor) or len(inputs) < 2:
                return
            residual = inputs[0]
            if not isinstance(residual, torch.Tensor):
                return
            # Use output - residual to capture the actual applied update (after dropout/alpha).
            # inputs[1] is the pre-dropout/pre-alpha update.
            update = output - residual
            res_norm = residual.norm(dim=-1).mean()
            upd_norm = update.norm(dim=-1).mean()
            ratio = (update.norm(dim=-1) / (residual.norm(dim=-1) + self.eps)).mean()

            self._log_metric(
                f"residual_update/activations/{name}/residual_norm",
                res_norm,
            )
            self._log_metric(
                f"residual_update/activations/{name}/update_norm",
                upd_norm,
            )
            self._log_metric(
                f"residual_update/activations/{name}/update_residual_ratio",
                ratio,
            )

        return hook

    def _make_residual_stream_backward_hook(self, name: str):
        def hook(
            module: nn.Module,
            grad_input: Tuple[Optional[torch.Tensor], ...],
            grad_output: Tuple[Optional[torch.Tensor], ...],
        ):
            del module, grad_output
            if not self._should_log():
                return
            if len(grad_input) < 2:
                return
            grad_residual = grad_input[0]
            grad_update = grad_input[1]
            if grad_residual is None or grad_update is None:
                return

            res_grad_norm = grad_residual.norm(dim=-1).mean()
            self._log_metric(
                f"residual_update/grads/{name}/residual_norm",
                res_grad_norm,
            )

        return hook

    def _make_activation_forward_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            del module, inputs
            if not self._should_log():
                return
            if not isinstance(output, torch.Tensor):
                return

            if self.track_activation_rmse or self.track_activation_norm or self.track_activation_meanvar:
                act = get_full_tensor(output.detach()).float()
                if self.track_activation_rmse:
                    self._log_metric(f"activations/{name}/rmse", self._vector_rmse(act))
                if self.track_activation_norm:
                    self._log_metric(f"activations/{name}/norm", self._mean_norm_over_batch_tokens(act))
                if self.track_activation_meanvar:
                    act_mean, act_std = self._vector_meanstd(act)
                    self._log_metric(f"activations/{name}/mean", act_mean)
                    self._log_metric(f"activations/{name}/stddev", act_std)

        return hook

    def _make_activation_backward_hook(self, name: str):
        def hook(
            module: nn.Module,
            grad_input: Tuple[Optional[torch.Tensor], ...],
            grad_output: Tuple[Optional[torch.Tensor], ...],
        ):
            del module, grad_input
            if not self._should_log():
                return
            if not grad_output or grad_output[0] is None or not isinstance(grad_output[0], torch.Tensor):
                return

            if self.track_activation_grad_rmse or self.track_activation_grad_norm or self.track_activation_grad_meanvar:
                act_grad = get_full_tensor(grad_output[0].detach()).float()
                if self.track_activation_grad_rmse:
                    self._log_metric(f"activation_grads/{name}/rmse", self._vector_rmse(act_grad))
                if self.track_activation_grad_norm:
                    self._log_metric(f"activation_grads/{name}/norm", self._mean_norm_over_batch_tokens(act_grad))
                if self.track_activation_grad_meanvar:
                    act_grad_mean, act_grad_std = self._vector_meanstd(act_grad)
                    self._log_metric(f"activation_grads/{name}/mean", act_grad_mean)
                    self._log_metric(f"activation_grads/{name}/stddev", act_grad_std)

        return hook

    def _make_layer_norm_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            del output
            if not self._should_log():
                return
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return

            x_full = get_full_tensor(x.detach()).float()
            if isinstance(module, RMSNorm):
                variance = x_full.pow(2).mean(-1)
            else:
                x_mean = x_full.mean(-1, keepdim=True)
                variance = (x_full - x_mean).pow(2).mean(-1)

            eps = getattr(module, "eps", None)
            if eps is None:
                return

            threshold = eps * (1.0 + self.eps_hit_tolerance)
            hit = (variance <= threshold).float().sum()

            self._log_metric(
                f"layernorm_eps/{name}/eps_hit",
                hit,
                reduce_type=ReduceType.sum,
                merge_strategy=MetricMergeStrategy.sum,
            )

        return hook

    def post_step(self):
        if not self._should_log():
            self._prev_params = None
            return

        model = getattr(self.trainer.train_module, "model", None)
        if model is None or not isinstance(model, nn.Module):
            self._prev_params = None
            return

        optim = getattr(self.trainer.train_module, "optim", None)
        if (
            self.track_update_param_ratio
            or self.track_update_rmse
            or self.track_param_movement
        ) and self._prev_params is not None:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                prev = self._prev_params.get(name)
                if prev is None:
                    continue
                current = get_local_tensor(p.detach()).float()
                update = current - prev

                if self.track_update_param_ratio:
                    denom = current.abs().clamp(min=self.eps)
                    ratio = (update.abs() / denom).mean()
                    self._log_metric(f"params/{name}/update_param_ratio", ratio)

                if self.track_update_rmse:
                    update_rmse = self._rmse(update)
                    self._log_metric(f"params/{name}/update_rmse", update_rmse)
                
                if self.track_param_movement:
                    rel_change = update.abs() / (prev.abs() + self.eps)
                    moving_frac = (rel_change > self.param_movement_threshold).float().mean()
                    self._log_metric(
                        f"params/{name}/moving_fraction_gt_rel_{self.param_movement_threshold}",
                        moving_frac,
                    )
                
        if self.track_param_meanvar:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                p_local = get_local_tensor(p.detach()).float()
                self._log_metric(f"params/{name}/mean", p_local.mean())
                self._log_metric(f"params/{name}/stddev", p_local.var(unbiased=False).sqrt())

        if self.track_optimizer_state_rmse_meanvar and optim is not None:
            for param, state in optim.state.items():
                name = None
                if self._param_name_map is not None:
                    name = self._param_name_map.get(id(param))
                if name is None:
                    name = f"param_{id(param)}"

                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is not None:
                    exp_avg_t = get_local_tensor(exp_avg.detach()).float()
                    rmse = self._rmse(exp_avg_t)
                    mean, std = self._meanstd(exp_avg_t)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_rmse", rmse)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_mean", mean)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_stddev", std)
                if exp_avg_sq is not None:
                    exp_avg_sq_t = get_local_tensor(exp_avg_sq.detach()).float()
                    rmse = self._rmse(exp_avg_sq_t)
                    mean, std = self._meanstd(exp_avg_sq_t)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_sq_rmse", rmse)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_sq_mean", mean)
                    self._log_metric(f"optimizer_state/{name}/exp_avg_sq_stddev", std)

        if self.track_embedding_usage:
            self._log_embedding_metrics()

        if (self.track_norm_weight_scale or self.track_norm_weight_dist) and model is not None:
            from olmo_core.distributed.utils import get_rank

            wandb_callback = None
            if self.track_norm_weight_dist:
                from .wandb import WandBCallback

                for cb in self.trainer.callbacks.values():
                    if isinstance(cb, WandBCallback) and cb.enabled:
                        wandb_callback = cb
                        break

            hist_data: Dict[str, Any] = {}
            for name, module in model.named_modules():
                if not isinstance(module, RMSNorm) or module.weight is None:
                    continue
                w = get_full_tensor(module.weight.detach()).float()
                if getattr(module, "one_plus_gamma", False):
                    w = 1.0 + w  # effective weight
                if self.track_norm_weight_scale:
                    self._log_metric(f"norm_weight_scale/{name}/rms", self._rmse(w))
                    self._log_metric(f"norm_weight_scale/{name}/abs_max", w.abs().max())
                if wandb_callback is not None:
                    hist_data[f"{self.namespace}/norm_weight_dist/{name}"] = (
                        wandb_callback.wandb.Histogram(w.cpu().numpy())
                    )

            if hist_data and wandb_callback is not None and get_rank() == 0:
                wandb_callback.wandb.log(hist_data, step=self.step)


        self._prev_params = None
        self._embedding_token_ids.clear()

    def _make_lm_head_hook(self):
        """Hook to track LM head logits statistics and probability mass during training."""
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output):
            del module, inputs
            if not self._should_log():
                return

            # Only track during training (when output is LMOutputWithLoss with a loss field).
            # During inference, output is raw logits tensor.
            if not hasattr(output, "loss"):
                return

            logits = None
            if hasattr(output, "logits"):
                logits = output.logits
            else:
                return

            if logits is None or not isinstance(logits, torch.Tensor):
                return

            # Logit statistics are valid on any shard (no cross-vocab dependencies).
            logits_local = get_local_tensor(logits.detach()).float()
            logits_flat_local = logits_local.view(-1, logits_local.shape[-1])
            self._log_metric("lm_head/logit_max", logits_flat_local.max(dim=-1)[0].mean())
            self._log_metric("lm_head/logit_min", logits_flat_local.min(dim=-1)[0].mean())
            self._log_metric("lm_head/logit_avg", logits_flat_local.mean())
            self._log_metric("lm_head/logit_std", logits_flat_local.std())

            # top1_mass and entropy require softmax over the full vocab. For
            # vocab-parallel TP, get_full_tensor() gathers the shards via DTensor
            # before computing softmax. For non-TP runs it is a no-op.
            logits_full = get_full_tensor(logits.detach()).float()
            logits_flat_full = logits_full.view(-1, logits_full.shape[-1])
            probs = torch.softmax(logits_flat_full, dim=-1)
            self._log_metric("lm_head/top1_mass", probs.max(dim=-1)[0].mean())
            self._log_metric("lm_head/entropy", -(probs * torch.log(probs + self.eps)).sum(dim=-1).mean())

        return hook

    def _make_embedding_forward_hook(self):
        """Hook to track embedding usage (which token IDs are used in forward pass)."""
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            del module, output
            if not self._should_log():
                return

            # The input to nn.Embedding is the token IDs tensor
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return

            token_ids = inputs[0]
            # Store token IDs for this batch
            self._embedding_token_ids.append(token_ids.detach())

        return hook
