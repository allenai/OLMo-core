"""Default-off PP1 experiment: one global count reduction per microbatch.

Rowwise EP additionally requires the separately guarded model/forward opt-in.
"""

import torch
import torch.distributed as dist

from olmo_core.ops import attach_auxiliary_loss


def finish_batched_router_aux(activation, records):
    """Attach the original per-layer auxiliary losses after a packed count reduction.

    ``records`` is a call-local sequence of (router, original auxiliary inputs).
    It must never be retained on a model or reused by another microbatch. Each
    router still applies its original normalization, loss and metric calculation.
    Extra live scores/logits and backward scheduling require explicit qualification.
    """
    if not records:
        return activation
    group = records[0][0].lb_process_group
    experts = records[0][0].num_experts
    if group is None:
        raise RuntimeError("Batched count reduction requires a process group")
    for router, aux in records:
        if (
            not router.global_load_balancing
            or router.lb_process_group is not group
            or router.num_experts != experts
            or router.tp_mesh is not None
            or router.cp_mesh is not None
            or aux is None
            or len(aux) != 5
        ):
            raise RuntimeError("Batched count reduction requires matching ordinary global routers")
    counts = torch.stack([aux[2].float() for _, aux in records])
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
    for index, (router, aux) in enumerate(records):
        # compute_aux_loss normalizes in-place. Give each row independent storage:
        # mutating another view of the packed matrix would invalidate saved tensors.
        reduced = counts[index].clone()
        loss = router.compute_aux_loss(*aux, reduced_global_counts=reduced)
        if loss is not None:
            activation = attach_auxiliary_loss(activation, loss)
    return activation


compiled_finish_batched_router_aux = torch.compile(finish_batched_router_aux, dynamic=False)
