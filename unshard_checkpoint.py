import argparse
import logging
from pathlib import Path

from olmo_core.distributed.checkpoint import unshard_checkpoint

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Unshard an FSDP checkpoint to allow loading with architecture changes"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="gs://ai2-llm/checkpoints/OLMo25-from476838/step500680",
        help="Path to the sharded checkpoint (local or GCS)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gs://ai2-llm/checkpoints/OLMo25-from476838/step500680-unsharded",
        help="Path to save the unsharded checkpoint",
    )
    parser.add_argument(
        "--use-safetensors",
        action="store_true",
        help="Save in safetensors format instead of PyTorch format",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists",
    )
    
    args = parser.parse_args()
    
    log.info(f"Unsharding checkpoint from: {args.input}")
    log.info(f"Saving unsharded checkpoint to: {args.output}")
    
    try:
        unshard_checkpoint(
            dir=args.input,
            target_dir=args.output,
            save_overwrite=args.overwrite,
            use_safetensors=args.use_safetensors,
        )
        log.info("Successfully unsharded checkpoint!")
        log.info(f"You can now load this checkpoint with:")
        log.info(f"  load_path='{args.output}'")
        log.info(f"  load_strategy=LoadStrategy.always")
        log.info(f"And the model will handle the architecture mismatch with strict=False")
        
    except Exception as e:
        log.error(f"Failed to unshard checkpoint: {e}")
        raise


if __name__ == "__main__":
    main()