import dataclasses
import math
from dataclasses import dataclass

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import (
    WSDS,
    OptimGroupOverride,
    Scheduler,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration

from .base import RunConfigurator
from .utils import format_count, format_tokens


@dataclass(kw_only=True)
class WSDSChinchillaRunConfigurator(RunConfigurator):
    """
    A run configurator that uses WSD-S learning rate scheduling and Chinchilla scaling laws.
    """

    chinchilla_multiple: float
    """
    How long to train each run for, expressed as a multiple of the Chinchilla-optimal duration
    which must be a power of 2.
    """
    decay_fraction: float = 0.1
    """The duration of each decay as a fraction of the period. Must be at least 10%."""

    def __post_init__(self):
        if self.chinchilla_multiple < 0.5 or not math.log(self.chinchilla_multiple, 2).is_integer():
            raise OLMoConfigurationError(
                "'chinchilla_multiple' must be at least 0.5 and a power of 2"
            )
        if not (0 < self.decay_fraction < 0.5):
            raise OLMoConfigurationError(
                "'decay_fraction' must be greater than 0.0 and less than 0.5"
            )

    def configure_duration(self, num_params: int) -> Duration:
        return Duration.chinchilla_tokens(
            self.chinchilla_multiple,
            model_params=num_params,
        )

    def configure_target_batch_size(self, num_params: int) -> int:
        # Calculate global batch size according to https://api.semanticscholar.org/CorpusID:270764838
        # which assumes a sequence length of 2048.
        return round(2048 * 160 * (num_params / 108_000_000) ** (2 / 3))

    def configure_optimizer(self, num_params: int) -> SkipStepAdamWConfig:
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        # but divide by 2 for WSD schedule (seems to work emperically).
        lr = 0.0047 * (num_params / 108_000_000) ** (-1 / 3)
        lr /= 2.0
        return SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(
                0.9,
                0.95,  # NOTE: paper above suggest using larger beta2 (~0.99) for small batch sizes.
            ),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        )

    def configure_chinchilla_periods(self, num_params: int) -> tuple[int, list[float]]:
        # Warm up 1 token per parameter according to https://api.semanticscholar.org/CorpusID:270764838
        warmup = num_params

        # Generate Chinchilla (decay) periods as multiples of two, but at least the minimum.
        chinchilla_periods: list[float] = []
        max_pow = math.log(self.chinchilla_multiple, 2)
        assert max_pow.is_integer()  # checked in `__post_init__()` as well.
        for p in range(-1, int(max_pow) + 1):
            period = 2**p
            chinchilla_periods.append(period)

        return warmup, chinchilla_periods

    def configure_lr_scheduler(self, num_params: int) -> Scheduler:
        warmup, chinchilla_periods = self.configure_chinchilla_periods(num_params)
        period_lengths = []
        for pidx, c in enumerate(chinchilla_periods):
            period = Duration.chinchilla_tokens(c, model_params=num_params).value
            if pidx == 0:
                period_lengths.append(period)
            else:
                period_lengths.append(
                    period
                    - Duration.chinchilla_tokens(
                        chinchilla_periods[pidx - 1], model_params=num_params
                    ).value
                )

        return WSDS(
            units=SchedulerUnits.tokens,
            warmup=warmup,
            decay_fraction=self.decay_fraction,
            period_lengths=period_lengths,
        )

    def configure_checkpoint_intervals(self, num_params: int) -> list[tuple[Duration, str]]:
        # We save two checkpoints for each period. One right before the decay and one at the bottom
        # of the decay (end of the period).
        _, chinchilla_periods = self.configure_chinchilla_periods(num_params)
        checkpoints: list[tuple[Duration, str]] = []
        for pidx, c in enumerate(chinchilla_periods):
            period = Duration.chinchilla_tokens(c, model_params=num_params)
            period_length: int
            if pidx == 0:
                period_length = period.value
            else:
                period_length = (
                    period.value
                    - Duration.chinchilla_tokens(
                        chinchilla_periods[pidx - 1], model_params=num_params
                    ).value
                )
            pre_decay = dataclasses.replace(
                period, value=period.value - round(period_length * self.decay_fraction)
            )
            checkpoints.append((pre_decay, f"Period {pidx+1}, {c}xC pre-decay"))
            checkpoints.append((period, f"Period {pidx+1}, {c}xC post-decay"))
        return checkpoints

    def plot_lr_schedule(
        self, num_params: int, *, batch_size: int | None = None, save_path: PathOrStr | None = None
    ):
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "matplotlib and pandas are required to use the plotting functionality."
            ) from exc

        if batch_size is None:
            batch_size = self.configure_target_batch_size(num_params)

        optim = self.configure_optimizer(num_params)
        scheduler = self.configure_lr_scheduler(num_params)
        warmup, chinchilla_periods = self.configure_chinchilla_periods(num_params)
        t_max = self.configure_duration(num_params).value
        tokens_seen = 0
        tokens = []
        lrs = []
        while tokens_seen < t_max:
            tokens_seen += batch_size
            lr = float(scheduler.get_lr(optim.lr, tokens_seen, t_max))
            tokens.append(tokens_seen)
            lrs.append(lr)

        df = pd.DataFrame({"tokens": tokens, "LR": lrs})
        df.plot(x="tokens", y="LR", legend=False, figsize=(12, 6))
        plt.grid(True)

        for c in chinchilla_periods:
            period = Duration.chinchilla_tokens(c, model_params=num_params).value
            plt.axvline(x=period, color="red", linestyle="--", alpha=0.5)
            plt.text(
                period,
                0.0,
                f"{c}xC",
                color="red",
                alpha=0.5,
                horizontalalignment="left",
                rotation=45,
            )

        # Plot checkpoint intervals.
        # But since checkpoint intervals are ultimately configured in steps, not tokens, we convert
        # to steps and then back to tokens to ensure accuracy.
        checkpoint_intervals = [
            batch_size * (d.value // batch_size)
            for d, _ in self.configure_checkpoint_intervals(num_params)
        ]
        plt.scatter(
            checkpoint_intervals,
            [float(scheduler.get_lr(optim.lr, t, t_max)) for t in checkpoint_intervals],
            color="green",
            label="Checkpoint",
        )

        plt.title(
            f"Learning rate schedule for a {format_count(num_params)} model out to {self.chinchilla_multiple}xC"
        )

        caption = (
            f"peak LR={optim.lr:.6f}, batch size={format_tokens(batch_size)}\n"
            f"warmup={format_tokens(warmup)} / {warmup // batch_size:,d} steps, "
            f"duration={format_tokens(t_max)} / {t_max // batch_size:,d} steps"
        )
        plt.xlabel(f"Tokens\n\n{caption}")

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
