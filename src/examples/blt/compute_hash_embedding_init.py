from argparse import ArgumentParser
from pathlib import Path
import traceback
from sklearn.decomposition import TruncatedSVD
import logging
from dataclasses import dataclass
from collections import Counter

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.blt.embed import byte_group_hash_function
from olmo_core.data import TokenizerConfig, ByteTokenizerConfig, NumpyDatasetConfig, NumpyDatasetType, NumpyDataLoaderConfig, DataCollator
from olmo_core.utils import prepare_cli_environment
from olmo_core.nn.transformer import TransformerConfig

log = logging.getLogger(__name__)

_DATA_SOURCES = open(Path(__file__).parent / "data_sources.txt").read().strip().splitlines()
DATA_PATHS = ["/weka/oe-training-default/" + x for x in _DATA_SOURCES]
SEQUENCE_LENGTH = 1024
BATCH_SIZE = 1024
N_BATCHES_TO_USE_FOR_ESTIMATE = 10000
NUM_WORKERS = 16
DATA_WORK_DIR = "/tmp/dataset-cache"

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/weka/oe-training-default/benjaminm/olmo_1b_blt_hash_embedding_init"),
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--hash_byte_group_size",
        type=list,
        default=[3, 4, 5, 6, 7, 8],
        help="List of byte group sizes for hashing. Default is [3, 4, 5, 6, 7, 8].",
    )
    parser.add_argument(
        "--hash_byte_group_vocab",
        type=int,
        default=100_002,
        help="Vocabulary size for byte group hashing. Default is 100002.",
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
        "--teacher_ckpt_path",
        type=Path,
        default=Path("/weka/oe-training-default/benjaminm/checkpoints/olmo2_1b/model_and_optim"),
        help="Path to the teacher model checkpoint.",
    )
    return parser.parse_args()

@dataclass
class CountCollator(DataCollator):
    def __call__(self, items):
        input_ids = [x["input_ids"] if isinstance(x, dict) else x for x in items]
        flat_input_ids = [item for sublist in input_ids for item in sublist.tolist()]

        return Counter(flat_input_ids)

def populate_embedding_matrix(
    embedding_matrix: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    token_init_mapping: dict[int, list[int]],
    token_counter: Counter,
):
    unset_indices = set(range(embedding_matrix.shape[0]))

    for byte_id, token_ids in tqdm(token_init_mapping.items(), desc="Populating embedding matrix"):
        total_count = sum(token_counter[token_id] for token_id in token_ids)

        for token_id in token_ids:
            embedding_matrix[byte_id] += teacher_embeddings[token_id] * token_counter[token_id] / total_count
        
        unset_indices.remove(byte_id)

    embedding_matrix[sorted(unset_indices)] = teacher_embeddings.mean(dim=0).unsqueeze(0)

    return embedding_matrix

def main():
    args = parse_args()

    tokenizer_config = TokenizerConfig.dolma2()
    byte_tokenizer_config = ByteTokenizerConfig.blt()

    dset = NumpyDatasetConfig(
        paths=DATA_PATHS,
        name=NumpyDatasetType.fsl,
        sequence_length=SEQUENCE_LENGTH, # subword sequence length
        tokenizer=tokenizer_config,
        work_dir=DATA_WORK_DIR,
    ).build()
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=BATCH_SIZE,
        seed=0,
        num_workers=NUM_WORKERS,
    ).build(dset, collator=CountCollator(pad_token_id=tokenizer_config.pad_token_id))
    data_loader.reshuffle()

    token_counter = Counter()
    for batch_idx, batch in tqdm(enumerate(data_loader._iter_batches()), total=N_BATCHES_TO_USE_FOR_ESTIMATE, desc="Counting tokens"):
        token_counter.update(batch)

        if batch_idx >= N_BATCHES_TO_USE_FOR_ESTIMATE:
            break

    for token_id in range(tokenizer_config.vocab_size):
        token_counter[token_id] += 1  # laplace smoothing

    model = TransformerConfig.olmo2_1B_v2(
        vocab_size=tokenizer_config.padded_vocab_size()
    ).build()
    load_model_and_optim_state(args.teacher_ckpt_path, model)
    teacher_embeddings = model.state_dict()["embeddings.weight"]

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

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)
    byte_tokenizer = byte_tokenizer_config.build()

    token_init_mapping = {}
    token_init_mapping[1] = {}  # for regular embeddings

    for token_id in tqdm(range(hf_tokenizer.vocab_size), desc="Mapping token ids to hash ids."):
        byte_ids = byte_tokenizer.patch_ids_to_byte_ids([token_id])

        for func_nb in range(args.hash_byte_group_nb_functions):
            for byte_group_size in args.hash_byte_group_size:
                if (byte_group_size, func_nb) not in token_init_mapping:
                    token_init_mapping[(byte_group_size, func_nb)] = {}

                hash_ids = byte_group_hash_function(
                    torch.tensor([byte_ids]),
                    byte_group_size,
                    hash_func_nb=func_nb,
                    max_hash=args.hash_byte_group_vocab,
                )[0]

                for hash_id in hash_ids.tolist():
                    if hash_id not in token_init_mapping[(byte_group_size, func_nb)]:
                        token_init_mapping[(byte_group_size, func_nb)][hash_id] = []
                    token_init_mapping[(byte_group_size, func_nb)][hash_id].append(token_id)

                for byte_id in byte_ids:
                    if byte_id not in token_init_mapping[1]:
                        token_init_mapping[1][byte_id] = []
                    token_init_mapping[1][byte_id].append(token_id)

    embedding_matrix = torch.zeros(
        (byte_tokenizer_config.padded_vocab_size(), args.local_d_model),
        dtype=torch.float32,
    )
    hash_embedding_matrices = [
        torch.zeros(
            (args.hash_byte_group_vocab, args.local_d_model),
            dtype=torch.float32,
        )
        for _ in range(args.hash_byte_group_nb_functions * len(args.hash_byte_group_size))
    ]

    populate_embedding_matrix(
        embedding_matrix,
        teacher_embeddings_reduced,
        token_init_mapping[1],
        token_counter,
    )

    i = 0
    for func_nb in range(args.hash_byte_group_nb_functions):
        for byte_group_size in args.hash_byte_group_size:
            populate_embedding_matrix(
                hash_embedding_matrices[i],
                teacher_embeddings_reduced,
                token_init_mapping[(byte_group_size, func_nb)],
                token_counter,
            )
            i += 1

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