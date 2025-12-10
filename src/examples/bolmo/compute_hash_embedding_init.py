from argparse import ArgumentParser
from pathlib import Path
import traceback
from sklearn.decomposition import TruncatedSVD
import logging
from dataclasses import dataclass, field
from typing import List, Optional, cast
import os
import pickle
import hashlib

import torch
from tqdm.auto import tqdm

from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.bolmo.embed import byte_group_hash_function
from olmo_core.data import TokenizerConfig, ByteTokenizer, ByteTokenizerConfig, NumpyDatasetConfig, NumpyDatasetType, NumpyDataLoaderConfig, DataCollator
from olmo_core.utils import prepare_cli_environment
from olmo_core.nn.transformer import TransformerConfig

log = logging.getLogger(__name__)

_DATA_SOURCES = open(Path(__file__).parent / "data_sources.txt").read().strip().splitlines()
DATA_PATHS = ["/weka/oe-training-default/" + x for x in _DATA_SOURCES]
SEQUENCE_LENGTH = 1024
BATCH_SIZE = 128
N_BATCHES_TO_USE_FOR_ESTIMATE = 1000
NUM_WORKERS = 128
DATA_WORK_DIR = "/tmp/dataset-cache"

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/weka/oe-training-default/benjaminm/olmo_1b_blt_hash_embedding_init_v2"),
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--hash_byte_group_size",
        type=str,
        default=",".join(str(x) for x in [3, 4, 5, 6, 7, 8]),
        help="List of byte group sizes for hashing. Default is [3, 4, 5, 6, 7, 8].",
    )
    parser.add_argument(
        "--hash_byte_group_vocab",
        type=str,
        default=",".join(str(x) for x in [100_002] * 6),
        help="Vocabulary sizes for byte group hashing. Default is 100002.",
    )
    parser.add_argument(
        "--hash_byte_group_nb_functions",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--local_d_model",
        type=int,
        default=1024,
        help="Dimensionality of the local model. Default is 1024.",
    )
    parser.add_argument(
        "--model_arch",
        default="olmo2_1B_v2",
        help="Model architecture to use. Default is 'olmo2_1B_v2'.",
    )
    parser.add_argument(
        "--teacher_ckpt_path",
        type=Path,
        default=Path("/weka/oe-training-default/benjaminm/checkpoints/olmo2_1b/model_and_optim"),
        help="Path to the teacher model checkpoint.",
    )
    return parser.parse_args()

@dataclass
class Collator(DataCollator):
    byte_tokenizer: Optional[ByteTokenizer] = None
    hash_byte_group_size: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8])
    hash_byte_group_vocab: List[int] = field(default_factory=lambda: [100_002] * 6)
    hash_byte_group_nb_functions: int = 1

    def __call__(self, items):
        assert self.byte_tokenizer is not None, "Byte tokenizer must be provided to the collator."

        all_input_ids = [x["input_ids"] if isinstance(x, dict) else x for x in items]
        token_init_mapping = {}
        token_init_mapping[1] = {}

        for original_input_ids in all_input_ids:
            original_input_ids = cast(torch.Tensor, original_input_ids).tolist()
            byte_ids, patch_lengths = self.byte_tokenizer.get_tokens_and_patch_lengths(original_input_ids, add_bos=False, skip_last=False)

            patch_ids = []
            for i, patch_length in enumerate(patch_lengths):
                patch_ids.extend([i] * patch_length)

            hash_embed_idx = 0
            for byte_group_size in self.hash_byte_group_size:
                for func_nb in range(self.hash_byte_group_nb_functions):
                    if (byte_group_size, func_nb) not in token_init_mapping:
                        token_init_mapping[(byte_group_size, func_nb)] = {}

                    hash_ids = byte_group_hash_function(
                        torch.tensor([byte_ids]),
                        byte_group_size,
                        hash_func_nb=func_nb,
                        max_hash=self.hash_byte_group_vocab[hash_embed_idx],
                    )[0]

                    for hash_id, patch_id in zip(hash_ids.tolist(), patch_ids):
                        token_id = original_input_ids[patch_id]

                        if hash_id not in token_init_mapping[(byte_group_size, func_nb)]:
                            token_init_mapping[(byte_group_size, func_nb)][hash_id] = {}
                        if token_id not in token_init_mapping[(byte_group_size, func_nb)][hash_id]:
                            token_init_mapping[(byte_group_size, func_nb)][hash_id][token_id] = 0

                        token_init_mapping[(byte_group_size, func_nb)][hash_id][token_id] += 1

                    hash_embed_idx += 1

            for byte_id, patch_id in zip(byte_ids, patch_ids):
                token_id = original_input_ids[patch_id]

                if byte_id not in token_init_mapping[1]:
                    token_init_mapping[1][byte_id] = {}
                if token_id not in token_init_mapping[1][byte_id]:
                    token_init_mapping[1][byte_id][token_id] = 0

                token_init_mapping[1][byte_id][token_id] += 1

        return token_init_mapping

def populate_embedding_matrix(
    embedding_matrix: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    token_init_mapping: dict[int, list[int]],
):
    unset_indices = set(range(embedding_matrix.shape[0]))

    for byte_id, token_id_map in tqdm(token_init_mapping.items(), desc="Populating embedding matrix"):
        total_count = sum(token_id_map[token_id] for token_id in token_id_map)

        for token_id in token_id_map:
            embedding_matrix[byte_id] += teacher_embeddings[token_id] * (token_id_map[token_id] / total_count)

        unset_indices.remove(byte_id)

    embedding_matrix[sorted(unset_indices)] = teacher_embeddings.mean(dim=0).unsqueeze(0)

    return embedding_matrix

def main():
    args = parse_args()

    hash_byte_group_size = [int(x) for x in args.hash_byte_group_size.split(",")]
    hash_byte_group_vocab = [int(x) for x in args.hash_byte_group_vocab.split(",")]

    tokenizer_config = TokenizerConfig.dolma2()
    byte_tokenizer_config = ByteTokenizerConfig.blt()

    byte_tokenizer = byte_tokenizer_config.build()

    dset = NumpyDatasetConfig(
        paths=DATA_PATHS,
        name=NumpyDatasetType.fsl,
        sequence_length=SEQUENCE_LENGTH, # subword sequence length
        tokenizer=tokenizer_config,
        work_dir=DATA_WORK_DIR,
    ).build()
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=BATCH_SIZE * SEQUENCE_LENGTH,
        seed=0,
        num_workers=NUM_WORKERS,
    ).build(dset, collator=Collator(
        pad_token_id=tokenizer_config.pad_token_id,
        byte_tokenizer=byte_tokenizer,
        hash_byte_group_size=hash_byte_group_size,
        hash_byte_group_vocab=hash_byte_group_vocab,
        hash_byte_group_nb_functions=args.hash_byte_group_nb_functions,
    ))
    data_loader.reshuffle()

    token_init_mapping = {}
    token_init_mapping[1] = {}

    arg_hash = hashlib.sha256()
    arg_hash.update(str(hash_byte_group_size).encode())
    arg_hash.update(str(hash_byte_group_vocab).encode())
    arg_hash.update(str(args.hash_byte_group_nb_functions).encode())
    arg_hash.update(str(args.teacher_ckpt_path).encode())
    arg_hash = arg_hash.hexdigest()

    cache_file = os.path.join(DATA_WORK_DIR, f"token_init_mapping_{arg_hash}.pickle")

    if os.path.exists(cache_file):
        log.info("Loading token init mapping from cache: %s", cache_file)
        token_init_mapping = pickle.load(open(cache_file, "rb"))
    else:
        log.info("Computing token init mapping...")
        for batch_idx, batch_token_init_mapping in tqdm(enumerate(data_loader._iter_batches()), total=N_BATCHES_TO_USE_FOR_ESTIMATE, desc="Computing token init mapping..."):
            # combine mappings
            for byte_group_size in hash_byte_group_size:
                for func_nb in range(args.hash_byte_group_nb_functions):
                    if (byte_group_size, func_nb) not in token_init_mapping:
                        token_init_mapping[(byte_group_size, func_nb)] = {}

                    for byte_id, token_id_map in batch_token_init_mapping[(byte_group_size, func_nb)].items():  # type: ignore
                        if byte_id not in token_init_mapping[(byte_group_size, func_nb)]:
                            token_init_mapping[(byte_group_size, func_nb)][byte_id] = {}

                        for token_id, count in token_id_map.items():
                            if token_id not in token_init_mapping[(byte_group_size, func_nb)][byte_id]:
                                token_init_mapping[(byte_group_size, func_nb)][byte_id][token_id] = 0

                            token_init_mapping[(byte_group_size, func_nb)][byte_id][token_id] += count
            
            for byte_id, token_id_map in batch_token_init_mapping[1].items():  # type: ignore
                if byte_id not in token_init_mapping[1]:
                    token_init_mapping[1][byte_id] = {}

                for token_id, count in token_id_map.items():
                    if token_id not in token_init_mapping[1][byte_id]:
                        token_init_mapping[1][byte_id][token_id] = 0

                    token_init_mapping[1][byte_id][token_id] += count

            if batch_idx >= N_BATCHES_TO_USE_FOR_ESTIMATE:
                break

        log.info("Saving token init mapping to cache: %s", cache_file)
        with open(cache_file, "wb") as f:
            pickle.dump(token_init_mapping, f)

    model = getattr(TransformerConfig, args.model_arch)(
        vocab_size=tokenizer_config.padded_vocab_size()
    ).build()
    load_model_and_optim_state(args.teacher_ckpt_path, model)
    teacher_embeddings = model.state_dict()["embeddings.weight"]

    if args.local_d_model != teacher_embeddings.shape[1]:
        assert args.local_d_model < teacher_embeddings.shape[1], "Local d_model must be less than or equal to global d_model."

        # need to reduce dimension to the local d_model to use as an init
        svd = TruncatedSVD(
            n_components=args.local_d_model,
            random_state=1234,
        ).fit(teacher_embeddings.cpu().numpy())
        log.info("SVD total explained variance ratio: %s", svd.explained_variance_ratio_.sum())

        teacher_embeddings_reduced = svd.transform(teacher_embeddings.cpu().numpy())
        # rescale to mean/std of the original embeddings
        mean = teacher_embeddings.mean(dim=0).mean()
        std = teacher_embeddings.std(dim=0).mean()
        teacher_embeddings_reduced = torch.tensor(teacher_embeddings_reduced, dtype=torch.float32)
        teacher_embeddings_reduced = (teacher_embeddings_reduced - teacher_embeddings_reduced.mean()) / teacher_embeddings_reduced.std() * std + mean
    else:
        teacher_embeddings_reduced = teacher_embeddings.cpu()

    embedding_matrix = torch.zeros(
        (byte_tokenizer_config.padded_vocab_size(), args.local_d_model),
        dtype=torch.float32,
    )
    hash_embedding_matrices = [
        torch.zeros(
            (hash_byte_group_vocab[hash_embed_idx], args.local_d_model),
            dtype=torch.float32,
        )
        for hash_embed_idx in range(args.hash_byte_group_nb_functions * len(hash_byte_group_size))
    ]

    populate_embedding_matrix(
        embedding_matrix,
        teacher_embeddings_reduced,
        token_init_mapping[1],
    )

    hash_embed_idx = 0
    for byte_group_size in hash_byte_group_size:
        for func_nb in range(args.hash_byte_group_nb_functions):
            populate_embedding_matrix(
                hash_embedding_matrices[hash_embed_idx],
                teacher_embeddings_reduced,
                token_init_mapping[(byte_group_size, func_nb)],
            )
            hash_embed_idx += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        embedding_matrix,
        args.output_dir / "embedding_init.pth",
    )
    for idx, hash_embedding_matrix in enumerate(hash_embedding_matrices):
        torch.save(
            hash_embedding_matrix,
            args.output_dir / f"hash_embedding_init_{idx}.pth",
        )

if __name__ == "__main__":
    prepare_cli_environment()
    try:
        main()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()
        import ipdb; ipdb.post_mortem()