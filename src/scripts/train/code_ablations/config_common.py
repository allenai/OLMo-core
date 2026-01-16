from olmo_core.data.source_mixture import SourceMixtureList, SourceMixtureConfig
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.transformer import TransformerConfig


S3_PREFIX = "s3://ai2-llm"
GS_PREFIX = "gs://ai2-llm"

OLMO_3_SEQUENCE_LENGTH = 8192
OLMO_3_MICROANNEAL_BATCH_SIZE = 2**21  # ~2M tokens
OLMO_3_MICROANNEAL_START_LR = 0.00020712352850360292
OLMO_3_MICROANNEAL_MAX_TOKENS = 10_000_000_000  # 10B tokens, microanneal
OLMO_3_MICROANNEAL_LOAD_PATH = f"{S3_PREFIX}/checkpoints/OLMo25/step1413814"

TOKENIZER_CONFIG = TokenizerConfig.dolma2()

ALL_DRESSED_50PCT_CONFIG = SourceMixtureConfig(
    source_name="all-dressed-snazzy2-v0.1-150b-50pct",
    target_ratio=0.5,
    paths=[
        f"{S3_PREFIX}/preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER_CONFIG.identifier}/all-dressed-snazzy2/*/*.npy"
    ],
)


STACK_EDU_50PCT_CONFIG = SourceMixtureConfig(
    source_name="stack-edu-python-v0.1-150b-50pct",
    target_ratio=0.5,
    paths=[f"{S3_PREFIX}/preprocessed/stack-edu/{TOKENIZER_CONFIG.identifier}/Python/*.npy"],
)


WEB_50PCT_STACK_EDU_PYTHON_50PCT_BASELINE_CONFIG = SourceMixtureList(
    sources=[ALL_DRESSED_50PCT_CONFIG, STACK_EDU_50PCT_CONFIG],
)

OLMO_3_7B_MODEL_CONFIG = TransformerConfig.olmo3_7B(vocab_size=TOKENIZER_CONFIG.padded_vocab_size())
