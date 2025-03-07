import argparse
import os
import numpy as np
from typing import List, Optional
from olmo_core.config import DType
import torch

from mup.coord_check import plot_coord_data
from mup import set_base_shapes
from mup import load_base_shapes as mup_load
import mup as mupo

from olmo_core.distributed.parallel import DataParallelType
from olmo_core.data import TokenizerConfig, NumpyFSLDataset, NumpyFSLDataLoader, DataCollator
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.scaling.coord_check import get_coord_data
from olmo_core.scaling.mup_utils import load_mu_model, save_base_shapes
from olmo_core.nn.functional import cross_entropy_loss

def get_dataloader(data_paths, batch_size: int):

    tokenizer_config = TokenizerConfig.dolma2()
    global_batch_size = batch_size

    dataset = NumpyFSLDataset(
        data_paths,
        sequence_length=4096,
        pad_token_id=tokenizer_config.pad_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        vocab_size=tokenizer_config.vocab_size,
        dtype=np.uint32,  
        metadata=None,
        include_instance_metadata=False,
        generate_doc_lengths=False,
        max_target_sequence_length=None  
    )

    assert global_batch_size % dataset.sequence_length == 0, \
        "Global batch size must be divisible by sequence length!"
    
    train_loader = NumpyFSLDataLoader(
        dataset,
        global_batch_size=global_batch_size,
        seed=34521,
        shuffle=True,
        num_workers=1,
        collator = DataCollator(pad_token_id=0),
        work_dir = dataset.work_dir
    )

    return train_loader


def coord_check(
    mup: bool,
    widths: List,
    config_path: str,
    batch_size: int,
    nsteps: int,
    nseeds: int,
    cuda: bool = False,
    output_dir: str = "",
    load_base_shapes: Optional[str] = None,
    legend: str = "brief",
    plot: bool = True,
):
    def model_generator(d_model, standparam=False):
        def f():
            base_shapes = mup_load(load_base_shapes)
            config = TransformerConfig.olmo2_190M(
                vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
                compile=True,
                d_model=d_model,
                dp_config=TransformerDataParallelConfig(
                    name=DataParallelType.fsdp,
                    param_dtype=DType.bfloat16,
                    reduce_dtype=DType.float32,
                    wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
                ),
            )
            config.use_mup = True
            device = torch.device('cuda')
            model = config.build(device=device)
            set_base_shapes(model, base_shapes)
            # print("✓ Successfully applied base shapes to model")
            # optimizer = mupo.optim.MuSGD(model.parameters(), lr=0.1)
            # print("✓ Successfully created muP optimizer")
            # print("\nChecking parameters for infshape attribute...")
            # missing_infshape = []
            # total_params = 0
            
            # for name, param in model.named_parameters():
            #     total_params += 1
            #     if not hasattr(param, 'infshape'):
            #         missing_infshape.append((name, param.shape))
            
            # if missing_infshape:
            #     print(f"✗ ISSUE: {len(missing_infshape)}/{total_params} parameters missing infshape attribute")
            #     print("\nSample parameters missing infshape:")
            #     for i, (name, shape) in enumerate(missing_infshape[:5]):
            #         print(f"  - {name}: {shape}")
            #     if len(missing_infshape) > 5:
            #         print(f"  ... and {len(missing_infshape) - 5} more")
            #     return False
            # else:
            #     print(f"✓ All {total_params} parameters have infshape attribute")                        
            return model
        return f

    optimizer = "adam"
    optimizer = optimizer.replace("mu", "")
    
    models = {width: model_generator(width, standparam=not mup) for width in widths}

    data_loader = get_dataloader(data_paths='sample-tokens.npy', batch_size=batch_size)
    if hasattr(data_loader, 'reshuffle'):
            data_loader.reshuffle()
    df = get_coord_data(
        models,
        data_loader,
        mup=True,
        # lr=lr,
        optimizer=optimizer,
        nseeds=nseeds,
        nsteps=nsteps,
        lossfn=cross_entropy_loss,
        cuda=cuda,
        compute_z_loss=True,
        show_progress=True,
    )

    prm = "mup" if mup else "sp"
    os.makedirs(output_dir, exist_ok=True)
    coords_file = os.path.join(output_dir, f"{prm}_olmo_{optimizer}_coord.csv")
    df.to_csv(coords_file, index=False)
    if plot:
        # Plot no more than 20 graphs
        step_interval = max(nsteps // 20, 1)
        df = df[df["t"] % step_interval == 0]
        df.loc[:, "t"] /= step_interval

        plot_coord_data(
            df,
            legend=legend,
            save_to=os.path.join(output_dir, f"{prm}_olmo_{optimizer}_coord.png"),
            suptitle=f"{prm} Transformer {optimizer} nseeds={nseeds}",
            face_color="xkcd:light grey" if not mup else None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run coord check for OLMo model with muP",
    )

    parser.add_argument("config_path")

    parser.add_argument("--save_base_shapes", type=str, default="", help="file location to save base shapes at")
    parser.add_argument("--load_base_shapes", type=str, default="", help="file location to load base shapes from")

    parser.add_argument("--batch_size", type=int, default=20, metavar="N", help="batch size")
    parser.add_argument("--widths", type=int, nargs="+", default=[2 ** i for i in range(4, 10)], help="widths to use for coord check")

    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument("--legend", type=str, help="'auto', 'brief', 'full', or False. This is passed to `seaborn.lineplot`.")

    parser.add_argument(
        "--coord_check",
        action="store_true",
        help="test μ parametrization is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.",
    )
    parser.add_argument("--coord_check_nsteps", type=int, default=3, help="Do coord check with this many steps.")
    parser.add_argument(
        "--coord_check_nseeds",
        type=int,
        default=3,
        help="number of seeds for testing correctness of μ parametrization",
    )

    parser.add_argument(
        "--coord_check_save_path",
        type=str,
        default="coord_checks",
        help="dir location for saving coord check plots",
    )

    args = parser.parse_args()
    print(args)

    if args.save_base_shapes:
        save_base_shapes(args.save_base_shapes)
        print("done and exit")
        import sys

        sys.exit()

    if args.coord_check:
        print("testing parametrization")

        os.makedirs(args.coord_check_save_path, exist_ok=True)
        import os
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29600"

        for use_mup in [True, False]:
            coord_check(
                mup=use_mup,
                widths=args.widths,
                config_path=args.config_path,
                batch_size=args.batch_size,
                nsteps=args.coord_check_nsteps,
                nseeds=args.coord_check_nseeds,
                cuda=args.cuda,
                output_dir=args.coord_check_save_path,
                legend=args.legend,
                load_base_shapes=args.load_base_shapes,
            )
