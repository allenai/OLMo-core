from typing import List, Optional

from mup.coord_check import _record_coords
from mup import set_base_shapes
from mup import load_base_shapes as mup_load
from olmo_core.data.utils import get_labels
import torch

def get_batch_loss(model, batch, lossfn, compute_z_loss):
    # TODO: move and import instead
    outputs = model(input_ids=batch["input_ids"])
    logits = outputs

    logits_for_loss = logits[..., :-1, :].contiguous()
    # shape: (batch_size * seq_len, vocab_size)
    logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
    # shape: (batch_size, seq_len)
    labels = get_labels(batch)
    # shape: (batch_size * seq_len,)
    labels = labels.view(-1)
    ce_loss, z_loss = lossfn(
        logits_for_loss,
        labels,
        ignore_index=-100,
        reduction="mean",
        compute_z_loss=compute_z_loss,
    )

    if compute_z_loss:
        assert z_loss is not None
        loss = ce_loss + z_loss
    else:
        loss = ce_loss
    return loss

def _get_coord_data(
    models,
    dataloader,
    optcls,
    mup,
    load_base_shapes: Optional[str] = None,
    nsteps=3,
    lossfn="xent",
    filter_module_by_name=None,
    fix_data=True,
    cuda=True,
    nseeds=1,
    output_fdict=None,
    input_fdict=None,
    param_fdict=None,
    show_progress=True,
    compute_z_loss=False,
):
    coordinates: List = []
    if fix_data:
        batch = next(iter(dataloader))
        dataloader = [batch] * nsteps
    if show_progress:
        from tqdm import tqdm

        pbar = tqdm(total=nseeds * len(models))

    for width, model_ in models.items():
        for i in range(nseeds):
            torch.manual_seed(i)
            model = model_()
            model = model.train()
            if cuda:
                model = model.cuda()
            if mup:
                base_shapes = mup_load(load_base_shapes)
                set_base_shapes(model, base_shapes, rescale_params=True)

            optimizer = optcls(model)
            for batch_idx, batch in enumerate(dataloader, 1):
                remove_hooks = []
                # add hooks
                for name, module in model.named_modules():
                    if filter_module_by_name and not filter_module_by_name(name):
                        continue
                    remove_hooks.append(
                        module.register_forward_hook(
                            _record_coords(
                                coordinates,
                                width,
                                name,
                                batch_idx,
                                output_fdict=output_fdict,
                                input_fdict=input_fdict,
                                param_fdict=param_fdict,
                            )
                        )
                    )
                if cuda:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.cuda()

                loss = get_batch_loss(model, batch, lossfn, compute_z_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # remove hooks
                for handle in remove_hooks:
                    handle.remove()

                if batch_idx == nsteps:
                    break
            if show_progress:
                pbar.update(1)
    if show_progress:
        pbar.close()
    import pandas as pd

    return pd.DataFrame(coordinates)


def get_coord_data(
    models, dataloader, load_base_shapes, mup, optimizer="adamw", lr=None, filter_trainable_by_name=None, **kwargs
):
    if lr is None:
        lr = 0.1 if optimizer == "sgd" else 1e-3
    if mup:
        print('get_coord_data muP: ', mup)
        from mup.optim import MuAdam as Adam
        from mup.optim import MuAdamW as AdamW
        from mup.optim import MuSGD as SGD
    else:
        from torch.optim import SGD, Adam, AdamW

    def get_trainable(model):
        params = model.parameters()
        if filter_trainable_by_name is not None:
            params = []
            for name, p in model.named_parameters():
                if filter_trainable_by_name(name):
                    params.append(p)
        return params

    if optimizer == "sgd":
        optcls = lambda model: SGD(get_trainable(model), lr=lr)  # noqa: E731
    elif optimizer == "adam":
        optcls = lambda model: Adam(get_trainable(model), lr=lr)  # noqa: E731
    elif optimizer == "adamw":
        optcls = lambda model: AdamW(get_trainable(model), lr=lr)  # noqa: E731
    elif optimizer is None:
        raise ValueError("optimizer should be sgd|adam|adamw or a custom function")

    data = _get_coord_data(models, dataloader, optcls, mup, load_base_shapes, **kwargs)
    data["optimizer"] = optimizer
    data["lr"] = lr
    return data