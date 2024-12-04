import math

from olmo_core.optim import (
    CosWithWarmup,
    ConstantScheduler,
    LinearScheduler,
    LinearWarmupDecoratorScheduler,
    InvSqrtScheduler,
    SequentialScheduler,
)


def test_constant_scheduler():
    initial_lr = 10.0
    max_steps = 10_000
    scheduler = ConstantScheduler()
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 2_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 10.0


def test_linear_scheduler():
    initial_lr = 10.0
    max_steps = 10_000
    alpha_f = 0.2
    scheduler = LinearScheduler(alpha_f=alpha_f)
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 2_000, max_steps) == 8.4
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 2.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) > scheduler.get_lr(initial_lr, 5_000, max_steps)


def test_inv_sqrt_scheduler():
    initial_lr = 10.0
    max_steps = 10_000
    alpha_f = 0.2
    step_offset = 1_000
    scheduler = InvSqrtScheduler(alpha_f=alpha_f, step_offset=step_offset)
    assert scheduler.get_lr(initial_lr, 1, max_steps) == 2.0 + 8.0 * math.sqrt(1_000.0 / 1_001.0)
    assert scheduler.get_lr(initial_lr, 2_000, max_steps) == 2.0 + 8.0 * math.sqrt(1_000.0 / 3_000.0)
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == scheduler.get_lr(initial_lr, 3_000, 10 * max_steps)


def test_linear_warmup_decorator_scheduler():
    initial_lr = 10.0
    warmup_min_lr = 4.0
    warmup_steps = 3_000
    max_steps = 10_000
    inner_scheduler = ConstantScheduler()
    scheduler = LinearWarmupDecoratorScheduler(
        inner=inner_scheduler, warmup_steps=warmup_steps, warmup_min_lr=warmup_min_lr
    )
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 4.0
    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == 6.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 10.0


def test_cos_with_warmup_scheduler():
    initial_lr = 10.0
    warmup_min_lr = 4.0
    warmup_steps = 3_000
    max_steps = 10_000
    alpha_f = 0.2
    scheduler = CosWithWarmup(warmup_steps=warmup_steps, alpha_f=alpha_f, warmup_min_lr=warmup_min_lr)
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 4.0
    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == 6.0
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) == 10.0
    assert scheduler.get_lr(initial_lr, 5_000, max_steps) == 2.0 + 8.0 * (1 + math.cos(math.pi * 2_000 / 7_000)) / 2
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 2.0


def test_sequential_scheduler():
    initial_lr = 10.0
    max_steps = 20_000

    first_scheduler = InvSqrtScheduler(alpha_f=0.5, step_offset=1_000)
    second_scheduler = CosWithWarmup()
    third_scheduler = LinearScheduler(t_max=3_000)
    schedulers_max_steps = [2_500, 4_000]
    scheduler = SequentialScheduler(
        schedulers=[first_scheduler, second_scheduler, third_scheduler], schedulers_max_steps=schedulers_max_steps
    )

    first_scheduler_final_lr = first_scheduler.get_lr(initial_lr, 2_500, 2_500)
    second_scheduler_final_lr = second_scheduler.get_lr(first_scheduler_final_lr, 4_000, 4_000)

    assert scheduler.get_lr(initial_lr, 1_000, max_steps) == first_scheduler.get_lr(initial_lr, 1_000, 2_500)
    assert scheduler.get_lr(initial_lr, 4_000, max_steps) == second_scheduler.get_lr(first_scheduler_final_lr, 1_500, 4_000)
    assert scheduler.get_lr(initial_lr, 7_500, max_steps) == third_scheduler.get_lr(second_scheduler_final_lr, 1_000, max_steps - 6_500)