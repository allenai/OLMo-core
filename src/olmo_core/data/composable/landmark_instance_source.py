import functools as ft
import hashlib
from dataclasses import dataclass

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from .instance_source import Instance, InstanceSource, InstanceSourceConfig


@dataclass
class LandmarkInstanceSourceConfig(InstanceSourceConfig):
    """
    Config for :class:`LandmarkInstanceSource`.

    :param source: The upstream instance source providing *content* instances (without landmark
        tokens). Its ``sequence_length`` must be a multiple of ``mem_freq``.
    :param mem_freq: The number of regular tokens between landmark tokens. The landmark block size
        is ``mem_freq + 1``.
    :param mem_id: The token ID to insert as the landmark token. This should be a reserved ID in
        the model's vocabulary whose embedding is learned during training.
    :param exclude_landmark_predictors: If ``True``, also exclude landmark tokens as loss
        *predictors* (not just as targets), matching reference implementations that drop landmark
        logits entirely. See :class:`LandmarkInstanceSource` for details. Defaults to ``False``.
    """

    source: InstanceSourceConfig
    mem_freq: int
    mem_id: int
    exclude_landmark_predictors: bool = False

    def build(self, work_dir: PathOrStr) -> "LandmarkInstanceSource":
        return LandmarkInstanceSource(
            self.source.build(work_dir),
            mem_freq=self.mem_freq,
            mem_id=self.mem_id,
            exclude_landmark_predictors=self.exclude_landmark_predictors,
            work_dir=work_dir,
        )


class LandmarkInstanceSource(InstanceSource):
    """
    An :class:`InstanceSource` that inserts a landmark token after every ``mem_freq`` tokens of an
    upstream (content) instance source, producing instances suitable for
    :class:`~olmo_core.nn.attention.LandmarkAttention`.

    Each upstream content instance of length ``C`` (a multiple of ``mem_freq``) is divided into
    ``C // mem_freq`` blocks of ``mem_freq`` tokens; a landmark token is appended to each block, so
    the emitted instance has length ``C // mem_freq * (mem_freq + 1)`` with landmark tokens at the
    fixed periodic positions ``pos % block_size == block_size - 1``.

    The emitted ``label_mask`` is ``False`` at landmark positions (so they are excluded from the
    loss) and otherwise preserves the upstream ``label_mask`` (defaulting to ``True`` for content
    tokens when the upstream source has none).

    .. note::
        With the standard left-shifted label convention, marking landmark positions in
        ``label_mask`` excludes the landmark tokens as prediction *targets*. By default, interior
        landmark positions are still used as predictors of the following content token; this differs
        slightly from reference implementations that drop landmark logits entirely. Set
        ``exclude_landmark_predictors=True`` to also drop the loss term *at* each landmark position
        (by masking the label of the following content token), so landmark tokens contribute to the
        loss neither as targets nor as predictors.
    """

    Config = LandmarkInstanceSourceConfig
    DISPLAY_ICON = ""

    def __init__(
        self,
        source: InstanceSource,
        *,
        mem_freq: int,
        mem_id: int,
        work_dir: PathOrStr,
        exclude_landmark_predictors: bool = False,
    ):
        if mem_freq < 1:
            raise OLMoConfigurationError(f"'mem_freq' must be >= 1 (got {mem_freq}).")
        block_size = mem_freq + 1
        if source.sequence_length % mem_freq != 0:
            raise OLMoConfigurationError(
                f"The upstream source 'sequence_length' ({source.sequence_length}) must be a "
                f"multiple of 'mem_freq' ({mem_freq})."
            )
        if source.max_sequence_length % mem_freq != 0:
            raise OLMoConfigurationError(
                f"The upstream source 'max_sequence_length' ({source.max_sequence_length}) must be "
                f"a multiple of 'mem_freq' ({mem_freq})."
            )
        super().__init__(
            sequence_length=source.sequence_length // mem_freq * block_size,
            max_sequence_length=source.max_sequence_length // mem_freq * block_size,
            work_dir=work_dir,
            label=source.label,
        )
        self._source = source
        self.mem_freq = mem_freq
        self.mem_id = mem_id
        self.block_size = block_size
        self.exclude_landmark_predictors = exclude_landmark_predictors

    @property
    def source(self) -> InstanceSource:
        return self._source

    @property
    def num_instances(self) -> int:
        return len(self._source)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"{self.mem_freq=},"
                f"{self.mem_id=},"
                f"{self.exclude_landmark_predictors=},"
                f"source={self.source.fingerprint},"
            ).encode()
        )
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return self.num_instances

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)
        instance = self._source[idx]
        input_ids = list(instance["input_ids"])
        label_mask = instance.get("label_mask")
        if label_mask is not None:
            label_mask = list(label_mask)

        # The content length is determined by the upstream source's current sequence length,
        # which may be smaller than this source's max during a sequence-length ramp.
        content_len = (len(input_ids) // self.mem_freq) * self.mem_freq

        mem_token = type(input_ids[0])(self.mem_id) if input_ids else self.mem_id

        new_ids: list = []
        new_mask: list = []
        for start in range(0, content_len, self.mem_freq):
            block = input_ids[start : start + self.mem_freq]
            new_ids.extend(block)
            new_ids.append(mem_token)
            if label_mask is not None:
                new_mask.extend(label_mask[start : start + self.mem_freq])
            else:
                new_mask.extend([True] * self.mem_freq)
            new_mask.append(False)  # landmark tokens are excluded from the loss (as targets)

        if self.exclude_landmark_predictors:
            # Also exclude landmark tokens as loss *predictors*. With left-shifted labels, the loss
            # term at a landmark position ``p`` predicts token ``p + 1`` (the next block's first
            # content token), so masking that token's ``label_mask`` drops the term. Landmark
            # positions are at ``p % block_size == block_size - 1``; mask each ``p + 1`` that exists.
            for p in range(self.block_size - 1, len(new_mask) - 1, self.block_size):
                new_mask[p + 1] = False

        return {"input_ids": new_ids, "label_mask": new_mask}

    def children(self):
        return [self.source]
