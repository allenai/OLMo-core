import math

from olmo_core.optim import (
    ConstantWithWarmup,
    CosWithWarmup,
    InvSqrtWithWarmup,
    LinearWithWarmup,
    SequentialScheduler,
)


def test_constant_with_warmup():
    initial_lr = 10.0
    warmup_min_lr = 4.0
    warmup_steps = 3_000
    max_steps = 10_000
    scheduler = ConstantWithWarmup(warmup_steps=warmup_steps, warmup_min_lr=warmup_min_lr)
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 4.0
    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == 6.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 5_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 10.0


def test_linear_with_warmup():
    initial_lr = 10.0
    warmup_min_lr = 4.0
    warmup_steps = 3_000
    max_steps = 10_000
    alpha_f = 0.2
    scheduler = LinearWithWarmup(
        warmup_steps=warmup_steps, alpha_f=alpha_f, warmup_min_lr=warmup_min_lr
    )
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 4.0
    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == 6.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 5_000, max_steps) == 10.0 - 8.0 * (2_000 / 7_000)
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 2.0


def test_inv_sqrt_with_warmup():
    initial_lr = 10.0
    warmup_min_lr = 4.0
    warmup_steps = 3_000
    max_steps = 10_000
    alpha_f = 0.2
    scheduler = InvSqrtWithWarmup(
        warmup_steps=warmup_steps, alpha_f=alpha_f, warmup_min_lr=warmup_min_lr
    )
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 4.0
    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == 6.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 5_000, max_steps) == 2.0 + 8.0 * math.sqrt(3_000 / 5_000)
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 2.0 + 8.0 * math.sqrt(3_000 / 10_000)


def test_cos_with_warmup_scheduler():
    initial_lr = 10.0
    warmup_min_lr = 4.0
    warmup_steps = 3_000
    max_steps = 10_000
    alpha_f = 0.2
    scheduler = CosWithWarmup(
        warmup_steps=warmup_steps, alpha_f=alpha_f, warmup_min_lr=warmup_min_lr
    )
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 4.0
    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == 6.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == 10.0
    assert (
        scheduler.get_lr(initial_lr, 5_000, max_steps)
        == 2.0 + 8.0 * (1 + math.cos(math.pi * 2_000 / 7_000)) / 2
    )
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 2.0


def test_sequential_scheduler():
    initial_lr = 10.0
    max_steps = 20_000

    first_scheduler = InvSqrtWithWarmup(alpha_f=0.5, warmup_steps=1_000)
    second_scheduler = CosWithWarmup(warmup_steps=0)
    third_scheduler = LinearWithWarmup(t_max=3_000, warmup_steps=0)
    schedulers_max_steps = [2_500, 4_000]
    scheduler = SequentialScheduler(
        schedulers=[first_scheduler, second_scheduler, third_scheduler],
        schedulers_max_steps=schedulers_max_steps,
    )

    first_scheduler_final_lr = first_scheduler.get_lr(initial_lr, 2_500, 2_500)
    second_scheduler_final_lr = second_scheduler.get_lr(first_scheduler_final_lr, 4_000, 4_000)

    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == first_scheduler.get_lr(
        initial_lr, 1_000, 2_500
    )
    assert scheduler.get_lr(initial_lr, 4_000, max_steps) == second_scheduler.get_lr(
        first_scheduler_final_lr, 1_500, 4_000
    )
    assert scheduler.get_lr(initial_lr, 7_500, max_steps) == third_scheduler.get_lr(
        second_scheduler_final_lr, 1_000, max_steps - 6_500
    )
