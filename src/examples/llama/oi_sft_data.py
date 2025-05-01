import numpy as np
from oi_data.data_transformation import (
    INPUT_IDS_KEY,
    LABELS_KEY,
    ATTENTION_MASK_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
transform_fn_args = [
    {"max_seq_length": 1024},
    {},
]
tc = TokenizerConfig(
    tokenizer_name_or_path="meta-llama/Llama-3.1-8B",
    chat_template_name="tulu",
    add_bos=True,
)
tokenizer = tc.tokenizer
dataset_mixer_list = ["allenai/tulu-3-sft-personas-algebra", "1.0"]
dataset_mixer_list_splits = ["train"]
dataset_transform_fn = ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
dataset_target_columns = TOKENIZED_SFT_DATASET_KEYS
dataset_cache_mode = "local"
dataset_local_cache_dir = "local_dataset_cache"
dataset_skip_cache = False
train_dataset = get_cached_dataset_tulu(
    dataset_mixer_list=dataset_mixer_list,
    dataset_mixer_list_splits=dataset_mixer_list_splits,
    tc=tc,
    dataset_transform_fn=dataset_transform_fn,
    transform_fn_args=transform_fn_args,
)
visualize_token(train_dataset[0][INPUT_IDS_KEY], tokenizer)

print("selecting 100 examples")
train_dataset = train_dataset.select(range(100)) # for debugging purposes

# create numpy arrays
token_ids = []
labels = []
attention_mask = []
for i in range(len(train_dataset)):
    token_ids.extend(train_dataset[i][INPUT_IDS_KEY])
    labels.extend(train_dataset[i][LABELS_KEY])
    attention_mask.extend(train_dataset[i][ATTENTION_MASK_KEY])

print("the maximum token id is", max(token_ids))
print("writing data to oi_dataset.npy")
token_ids_mmap = np.memmap("oi_token_ids.npy", mode="w+", dtype=np.uint32, shape=(len(token_ids),))
token_ids_mmap[:] = token_ids
token_ids_mmap.flush()
labels_mmap = np.memmap("oi_labels.npy", mode="w+", dtype=np.int32, shape=(len(labels),))
labels_mmap[:] = labels
labels_mmap.flush()
attention_mask_mmap = np.memmap("oi_attention_mask.npy", mode="w+", dtype=np.int32, shape=(len(attention_mask),))
attention_mask_mmap[:] = attention_mask
attention_mask_mmap.flush()
print("done")