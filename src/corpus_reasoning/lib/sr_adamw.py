"""Stochastic-rounding AdamW for olmo-core, applied by MONKEYPATCH (no edits to vendored OLMo-core).

Why: full-bf16 training keeps master weights in bf16. AdamW's weight update `p.add_(update)` then
rounds the tiny `lr*update` (e.g. 1e-5 * small) below bf16's 8-bit mantissa to ZERO under
round-to-nearest, so the weights barely move -> large accuracy loss (we measured 49.8% -> 27.2%
on NQ-k50). Stochastic rounding (SR) fixes this: compute the update in fp32 and round the result
to bf16 *probabilistically* (round up with probability proportional to the truncated remainder),
so updates survive in expectation across steps. This is the standard remedy for bf16 master weights;
olmo-core ships no SR, so we patch it in at runtime.

Usage (in the training process, before trainer.fit()):
    from corpus_reasoning.lib.sr_adamw import enable_stochastic_rounding
    enable_stochastic_rounding()

This swaps olmo_core.optim.adamw.{adamw_step, foreach_adamw_step} for SR-aware versions. They only
stochastically round bf16 params; fp32 params fall through to the exact same math as upstream, so
patching is a no-op for the baseline / bf16_moments arms even if accidentally left enabled.
"""

import torch

import olmo_core.optim.adamw as _adamw

# --- The bit-layout fact that makes this whole trick work -----------------------------------------
# fp32 and bf16 use the SAME 1 sign + 8 exponent bits; they differ ONLY in mantissa width:
#   fp32 = [ sign | 8 exp | 23 mantissa ]   (32 bits total)
#   bf16 = [ sign | 8 exp |  7 mantissa ]   (16 bits total)
# So a bf16 number is exactly the TOP 16 bits of the fp32 bit-pattern, and the BOTTOM 16 bits of an
# fp32 value are precisely the part bf16 cannot represent ("the remainder" we must round).
# Dropping the bottom 16 bits == truncation (round toward zero); we want stochastic rounding instead.
#
# _BF16_MASK keeps the top 16 bits and zeros the bottom 16. As an unsigned mask that's 0xFFFF0000;
# Python ints are signed and torch wants a same-dtype scalar, so we write it as the signed-int32
# value with that bit pattern: 0xFFFF0000 == -65536 in two's-complement int32.
_BF16_MASK = -65536


def stochastic_round_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """Stochastically round an fp32 tensor to bf16 (unbiased: E[round(x)] == x).

    Idea: the bottom 16 bits of x's fp32 pattern are the value bf16 must throw away. If we add a
    uniform random 16-bit integer to those bits BEFORE truncating, the addition carries up into the
    lowest bf16 mantissa bit with probability == (remainder / 2**16) -- i.e. we round UP exactly as
    often as the value sits toward the next bf16 grid point, and round DOWN otherwise. Averaged over
    many updates this preserves sub-mantissa increments that round-to-nearest would discard.
    """
    assert x.dtype == torch.float32, x.dtype
    # Reinterpret the fp32 bits as int32 WITHOUT changing the bits (a bit-cast, not a numeric cast).
    # .contiguous() because .view(dtype) requires a contiguous buffer.
    xi = x.contiguous().view(torch.int32)
    # One uniform random integer in [0, 2**16) per element -- this is the "dither" added to the bits
    # bf16 will drop. Drawn on the same device so there's no host<->device copy in the hot loop.
    noise = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int32)
    # 1) xi + noise: adding the dither can carry out of the low 16 bits into bit 16 (the least
    #    significant bf16 mantissa bit) -> that carry IS the probabilistic "round up".
    # 2) .bitwise_and_(_BF16_MASK): zero the low 16 bits = truncate to a bf16-aligned fp32 pattern.
    # 3) .view(torch.float32): reinterpret those bits back as a float (low bits now zero -> exact).
    # 4) .to(torch.bfloat16): narrow to bf16; lossless now since the value is already bf16-aligned.
    return (xi + noise).bitwise_and_(_BF16_MASK).view(torch.float32).to(torch.bfloat16)


def _sr_param_update(p: torch.Tensor, update: torch.Tensor, wd_scale) -> None:
    """Apply one AdamW weight step in-place:  p <- (p * wd_scale) + update.

    wd_scale folds in decoupled weight decay (== 1 - step_factor*lr*weight_decay).
    For a bf16 master we do the whole arithmetic in fp32 and round the *result* once with SR; for an
    fp32 master there's no rounding problem, so we use the exact upstream ops (and skip SR entirely).
    """
    if p.dtype == torch.bfloat16:
        # p.float() upcasts the CURRENT bf16 weight to fp32 (lossless). We then do weight decay and
        # the update in full fp32 precision so no information is lost MID-computation...
        p32 = p.float()
        p32.mul_(wd_scale)
        p32.add_(update.float())
        # ...and only at the very end collapse back to bf16, stochastically. This is the one place
        # rounding happens, and SR makes it unbiased so tiny updates aren't systematically dropped.
        p.copy_(stochastic_round_to_bf16(p32))
    else:  # fp32 master -> no sub-mantissa loss, so just do exactly what upstream AdamW does.
        p.mul_(wd_scale)
        p.add_(update)


def adamw_step_sr(p, grad, *, lr, betas, eps, weight_decay, exp_avg, exp_avg_sq, step,
                  step_factor, step_increment_bugfix=True):
    """SR drop-in for olmo_core.optim.adamw.adamw_step (single-tensor path).

    Mirrors upstream adamw_step EXACTLY except the final weight write goes through _sr_param_update.
    `step_factor` is olmo-core's skip-step gate (1.0 = take the step, 0.0 = skip this batch because
    its grad norm was an outlier) -- it multiplies every in-place change so a skipped step is a no-op.
    """
    beta1, beta2 = betas
    # --- first/second moment EMAs. Left in their stored dtype (bf16 here): moments are an average,
    # so low precision adds a little noise but no systematic bias -- the master weight is what's
    # sensitive, not these. Upstream math, unchanged.
    exp_avg.lerp_(grad.type_as(exp_avg), (step_factor * (1 - beta1)).type_as(exp_avg))     # m = (1-b1)*g + b1*m
    exp_avg_sq.mul_(1 - step_factor * (1 - beta2))                                          # v *= b2
    exp_avg_sq.add_(step_factor * grad * grad, alpha=1 - beta2)                             # v += (1-b2)*g^2
    # --- bias correction for the zero-initialized EMAs (standard Adam). step is this param's count.
    bias_correction1 = 1 - beta1 ** (step + 1)
    bias_correction2 = 1 - beta2 ** (step + 1)
    step_size = lr / bias_correction1
    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
    # --- the raw Adam step:  update = -lr_eff * m_hat / (sqrt(v_hat) + eps), gated by step_factor.
    update = -step_size * torch.div(exp_avg, denom)
    update.mul_(step_factor)
    # --- write it into the (bf16) master with SR. Upstream applies weight decay to p first and then
    # adds the update; since both touch only p (the moments above don't depend on p), folding them
    # into a single "p*wd_scale + update" fp32 expression and rounding once is mathematically the
    # same -- and avoids rounding p twice.
    _sr_param_update(p, update, 1 - step_factor * (lr * weight_decay))
    if step_increment_bugfix:
        step.add_(step_factor)  # advance this param's step counter (also gated, so skips don't count)


def foreach_adamw_step_sr(params, grads, exp_avgs, exp_avg_sqs, steps, *, lr, betas, eps,
                          weight_decay, step_factor, step_increment_bugfix=True):
    """SR drop-in for olmo_core.optim.adamw.foreach_adamw_step (multi-tensor / "foreach" path).

    Same Adam math as adamw_step_sr, but applied to the WHOLE param group at once with torch's
    _foreach_* multi-tensor kernels (one fused launch over a list of tensors -> faster than a Python
    loop). Only the final weight write drops back to a per-tensor loop, because the fp32-temp + SR
    can't be expressed as a single foreach op.
    """
    if not params:
        return  # group had no params with grads
    beta1, beta2 = betas
    # Cast each grad to its moment's dtype up front (matches upstream; keeps the foreach ops dtype-uniform).
    grads = [g.type_as(ea) for g, ea in zip(grads, exp_avgs)]

    # --- first moment EMA: m = (1-w1)*m + w1*g, done as two foreach ops (upstream avoids foreach_lerp_
    # because it misbehaves with DTensor). w1 = step_factor*(1-beta1) gates the update.
    w1 = step_factor * (1 - beta1)
    torch._foreach_mul_(exp_avgs, 1.0 - w1)
    torch._foreach_add_(exp_avgs, torch._foreach_mul(grads, w1))

    # --- second moment EMA: v = (1-w2)*v + w2*g^2
    grad_squares = torch._foreach_mul(grads, grads)
    w2 = step_factor * (1 - beta2)
    torch._foreach_mul_(exp_avg_sqs, 1.0 - w2)
    torch._foreach_add_(exp_avg_sqs, torch._foreach_mul(grad_squares, w2))

    # --- per-param bias corrections (each param may be at a different step count -> stack to a vector).
    steps_t = torch.stack(steps)
    bias_corrections1 = 1 - torch.pow(beta1, steps_t + 1)
    bias_corrections2 = 1 - torch.pow(beta2, steps_t + 1)
    step_sizes = lr / bias_corrections1

    # --- denom = sqrt(v)/sqrt(bias2) + eps  (built in place across the list; .unbind() turns the
    # length-N correction vector into N scalars so the foreach op pairs one per tensor).
    denoms = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denoms, bias_corrections2.sqrt().unbind())
    torch._foreach_add_(denoms, eps)

    # --- update_i = -step_factor * step_size_i * (m_i / denom_i)
    updates = torch._foreach_div(exp_avgs, denoms)
    torch._foreach_mul_(updates, (-step_factor * step_sizes).unbind())

    # --- weight write with SR, one tensor at a time (the only part that isn't a foreach kernel).
    # wd_scale is a scalar tensor shared by every param.
    wd_scale = 1 - step_factor * (lr * weight_decay)
    for p, u in zip(params, updates):
        _sr_param_update(p, u, wd_scale)

    if step_increment_bugfix:
        torch._foreach_add_(steps, [step_factor] * len(steps))  # bump each param's step counter


# Stash for the original (unpatched) functions so disable_stochastic_rounding() can restore them.
_ORIG = {}


def enable_stochastic_rounding() -> None:
    """Patch olmo-core's AdamW step functions with the SR versions. Idempotent.

    WHY this works as a monkeypatch: SkipStepAdamW._step / _step_foreach call the *module-level*
    names `adamw_step(...)` / `foreach_adamw_step(...)`. Python resolves those globals at CALL time,
    so rebinding the attributes on the olmo_core.optim.adamw module here makes every later optimizer
    step dispatch into our SR versions -- without touching any vendored OLMo-core file. Must be
    called in the training process before the first optimizer step (we do it from olmo_train.py).
    """
    if not _ORIG:  # remember the originals once, so repeated calls stay idempotent
        _ORIG["adamw_step"] = _adamw.adamw_step
        _ORIG["foreach_adamw_step"] = _adamw.foreach_adamw_step
    _adamw.adamw_step = adamw_step_sr
    _adamw.foreach_adamw_step = foreach_adamw_step_sr


def disable_stochastic_rounding() -> None:
    """Restore upstream behavior (mainly for tests)."""
    if _ORIG:
        _adamw.adamw_step = _ORIG["adamw_step"]
        _adamw.foreach_adamw_step = _ORIG["foreach_adamw_step"]
