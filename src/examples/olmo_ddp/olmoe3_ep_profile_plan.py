"""Pure, bounded topology choices for the final64GPU small-model EP comparison."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EPProfileTopology:
    """Keep dense DP/global batch fixed while folding node-local EP over its ranks."""

    ep: int = 1
    microbatch: int = 4
    gpus: int = 64
    sequence_length: int = 8192
    global_batch_tokens: int = 16 * 1024 * 1024

    def __post_init__(self):
        if self.gpus != 64 or self.ep not in (1, 2, 4, 8):
            raise ValueError("EP profiling requires64GPUs and node-local EP1/2/4/8")
        if self.microbatch not in (4, 8):
            raise ValueError(
                "Only the approved MB4 control and conditional MB8 probe are supported"
            )
        if self.sequence_length != 8192 or self.global_batch_tokens != 16 * 1024 * 1024:
            raise ValueError("EP profiling must preserve sequence8192 and16Mi tokens/update")
        if self.global_batch_tokens % (self.gpus * self.microbatch * self.sequence_length):
            raise ValueError("Global batch must be divisible by the microbatch wave")

    @property
    def accumulation(self):
        return self.global_batch_tokens // (self.gpus * self.microbatch * self.sequence_length)

    @property
    def expert_dp(self):
        return self.gpus // self.ep

    @classmethod
    def from_test_label(cls, label):
        """Parse explicit EP probes; unrelated existing comparisons retain their old topology."""
        if not label.startswith("ep"):
            return cls()
        fields = label.split("-")
        if len(fields) != 2 or not fields[1].startswith("mb"):
            raise ValueError(f"Invalid EP test label: {label}")
        return cls(ep=int(fields[0][2:]), microbatch=int(fields[1][2:]))

    def validate_rank_groups(self, groups, rank_to_node):
        """Fail closed if a runtime EP group spans physical nodes."""
        flattened = [rank for group in groups for rank in group]
        if sorted(flattened) != list(range(self.gpus)):
            raise ValueError("EP rank groups must partition all64ranks exactly once")
        for group in groups:
            if len(group) != self.ep or len({rank_to_node[rank] for rank in group}) != 1:
                raise ValueError(f"EP group must have{self.ep} ranks on one node: {group}")
