"""CPU regression of the actual restore audit using independently partitioned tensors."""

import ast
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import olmoe3_adaptive_decay_plan as plan
import torch


def check():
    """Require all four source shards, unchanged replicated state and loader position."""
    import hashlib

    source = Path(__file__).with_name("olmoe3_adaptive_decay_train.py")
    tree = ast.parse(source.read_text())
    functions = [
        node
        for node in tree.body
        if getattr(node, "name", None) in ("fingerprint", "verify_initial_reshard")
    ]
    assert len(functions) == 2
    rank = 0
    p = SimpleNamespace(SOURCE_GPUS=128, GPUS=32, START=plan.START, BATCH=plan.BATCH)
    adapter = SimpleNamespace(torch=torch, equal=lambda a, b: a == b)
    scope = dict(
        hashlib=hashlib,
        json=json,
        Path=Path,
        p=p,
        adapter=adapter,
        get_rank=lambda: rank,
        get_world_size=lambda: p.GPUS,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    fingerprint = scope["fingerprint"]
    verify = scope["verify_initial_reshard"]
    # Partition full reference tensors independently using torch.chunk.
    full = {
        "exp_avg": torch.arange(128 * 13, dtype=torch.float32),
        "exp_avg_sq": torch.arange(128 * 17, dtype=torch.float32) / 3,
    }
    replicated = {
        "optim_step": torch.tensor(plan.START),
        "buffer": torch.arange(5, dtype=torch.int64),
        "model_param/weight": torch.arange(31, dtype=torch.float32),
    }
    state = dict(
        step=plan.START,
        tokens=plan.START * plan.BATCH,
        loss_history=[2.4, 2.3],
        norm_history=[0.4, 0.3],
    )
    loader = dict(tokens_processed=state["tokens"], global_batch_size=plan.BATCH)
    with tempfile.TemporaryDirectory() as temporary:
        p.SOURCE = Path(temporary)
        (p.SOURCE / "resume_audit").mkdir()
        (p.SOURCE / "train").mkdir()
        for old_rank in range(128):
            tensors = dict(replicated)
            tensors.update({key: value.chunk(128)[old_rank] for key, value in full.items()})
            row = dict(
                state,
                gpus=128,
                rank=old_rank,
                tensors={key: fingerprint(value) for key, value in tensors.items()},
            )
            (p.SOURCE / "resume_audit" / f"rank{old_rank}.json").write_text(json.dumps(row))
        for rank in range(32):
            torch.save({"data_loader": loader}, p.SOURCE / "train" / f"rank{rank}.pt")
            states = {key: value.chunk(32)[rank].clone() for key, value in full.items()}
            states["optim_step"] = replicated["optim_step"]
            tm = SimpleNamespace(
                ep_enabled=False,
                pp_enabled=False,
                optim=SimpleNamespace(states=states),
                _persistent_model_buffer_state_dict=lambda: {"buffer": replicated["buffer"]},
                model=SimpleNamespace(
                    named_parameters=lambda: [("weight", replicated["model_param/weight"])]
                ),
            )

            def sample(trainer):
                tensors = dict(trainer.train_module.optim.states)
                tensors.update(replicated)
                return dict(
                    state, tensors={key: fingerprint(value) for key, value in tensors.items()}
                )

            adapter.hero = SimpleNamespace(state_sample=sample)
            trainer = SimpleNamespace(
                train_module=tm,
                data_loader=SimpleNamespace(
                    state_dict=lambda: dict(loader), tokens_processed=state["tokens"]
                ),
            )
            proof = verify(trainer, p.SOURCE)
            assert proof["verified_old_shard_pieces"] == 8
            assert proof["source_shards_per_current_rank"] == 4

        def rejects(mutate, restore):
            mutate()
            try:
                verify(trainer, p.SOURCE)
            except AssertionError:
                pass
            else:
                raise AssertionError("Corrupted restore passed")
            finally:
                restore()

        # Corruption in the fourth old shard must not escape a two-half audit.
        last = states["exp_avg"][-1].item()
        rejects(
            lambda: states["exp_avg"][-1].fill_(-999),
            lambda: states["exp_avg"][-1].fill_(last),
        )
        correct = states["exp_avg_sq"].clone()
        rejects(
            lambda: states["exp_avg_sq"].copy_(correct.roll(17)),
            lambda: states["exp_avg_sq"].copy_(correct),
        )
        rejects(
            lambda: setattr(trainer.data_loader, "state_dict", lambda: {"bad_offset": 1}),
            lambda: setattr(trainer.data_loader, "state_dict", lambda: dict(loader)),
        )
    result = dict(
        passed=True,
        source_gpus=128,
        target_gpus=32,
        ranks_checked=32,
        source_shards_per_current_rank=4,
        fourth_piece_corruption_rejected=True,
        reordered_shards_rejected=True,
        changed_loader_rejected=True,
        scope="CPU audit logic; actual checkpoint restore remains a GPU startup gate",
    )
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    check()
