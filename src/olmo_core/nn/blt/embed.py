import torch
from torch import nn

primes = [
    1000000007,
    5915587277,
    1500450271,
    3267000013,
    5754853343,
    4093082899,
    9576890767,
    3628273133,
    2860486313,
    5463458053,
    3367900313,
]


def rolling_polynomial_hash(t, hash_func_nb: int = 0):
    # DIVERGENCE FROM BLT: avoid sync
    prime_powers = primes[hash_func_nb] ** torch.arange(t.shape[-1], dtype=torch.int64, device=t.device)
    return torch.sum(t * prime_powers, dim=-1)


def byte_group_hash_function(
    x: torch.Tensor, group_size: int = 2, hash_func_nb: int = 0, max_hash: int = 30000
):
    """
    Returns a hash of the input x and maps it to a value in the range [0, max_hash].

    expects: x of shape (batch_size, seq_len) with values as ids in the token vocab.
    returns a tensor  of shape (batch_size, seq_len) with values in the range [0, max_hash].

    Note: max hash can make a big difference on the number of collisions.
    """
    with torch.no_grad():
        bs, seq_len = x.shape

        prefix = torch.zeros(bs, group_size - 1, dtype=torch.int64, device=x.device)
        x = torch.cat([prefix, x], dim=1)
        windows = x.unfold(1, group_size, 1)
        hashes = rolling_polynomial_hash(windows, hash_func_nb)
        hash_values_range = hashes % max_hash
    hash_values_range.requires_grad = False
    return hash_values_range


def add_hash_embeddings(
    embeddings: torch.Tensor,
    tokens: torch.Tensor,
    encoder_hash_tok_embeddings: nn.ModuleList,
    encoder_hash_byte_group_nb_functions: int,
    encoder_hash_byte_group_size: list,
    encoder_hash_byte_group_vocab: list,
) -> torch.Tensor:
    """
    Compute embeddings using hash token embeddings.

    Args:
        embeddings: Input embeddings tensor of shape (batch_size, seq_len, d_model)
        tokens: Input tokens tensor
        encoder_hash_tok_embedding: ModuleList of hash token embeddings
        encoder_hash_byte_group_nb_functions: Number of hash functions
        encoder_hash_byte_group_size: List of byte group sizes
        encoder_hash_byte_group_vocab: Vocabulary size for hash embeddings

    Returns:
        torch.Tensor: Embeddings tensor augmented with hash token embeddings, shape (batch_size, seq_len, d_model)
    """
    out_embeddings = embeddings

    hash_embed_idx = 0
    for byte_group_size in encoder_hash_byte_group_size:
        for func_nb in range(encoder_hash_byte_group_nb_functions):
            hash_ids = byte_group_hash_function(
                tokens,
                byte_group_size,
                hash_func_nb=func_nb,
                max_hash=encoder_hash_byte_group_vocab[hash_embed_idx],
            )
            hash_tok_embedding = encoder_hash_tok_embeddings[hash_embed_idx]
            out_embeddings = out_embeddings + hash_tok_embedding(hash_ids)
            hash_embed_idx += 1

    assert hash_embed_idx == len(encoder_hash_tok_embeddings)
    return out_embeddings