import sys
from datetime import datetime
from pathlib import Path

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.internal import cookbook
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.experiment import (
    CliContext,
    ExperimentConfig,
    get_beaker_username,
)
from olmo_core.internal.experiment import (
    main as olmo_core_main,
)
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.scheduler import LinearWithWarmup, SchedulerUnits
from olmo_core.train import Duration
from olmo_core.train.train_module import TransformerTrainModuleConfig

# Change these to match the config you want to use
SEQ_LENGTH = 8192
GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens
MAX_TOKENS = 10_000_000_000  # 10B
LR = 0.00020712352850360292 / 2  # halfing the LR
SEED = 1337
TOKENIZER_CONFIG = TokenizerConfig.dolma2()
PRIORITY = "high"
WORKSPACE = "ai2/olmo4"
BUDGET = "ai2/oe-base"
NUM_NODES = 4
LOAD_PATH = "gs://ai2-llm/checkpoints/OLMo25/step1413814"
BASE_SAVE_DIR = "s3://ai2-llm/checkpoints"
COMMIT_PACK_PREFIX = "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned"
DOLMA_3_PREFIX = "s3://ai2-llm/preprocessed/dolma2-0625/v0.1-150b"

MODEL_CONFIG = TransformerConfig.olmo3_7B(vocab_size=TOKENIZER_CONFIG.padded_vocab_size())


COMMIT_PACK_PATHS = [
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/bluespec/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/bluespec/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/bluespec/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/bluespec/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/bluespec/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/bluespec/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/bluespec/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/bluespec/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/bluespec/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/bluespec/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/c/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/c/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/c/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/c/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/c/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/c/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/c/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/c/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/c/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/c/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/clojure/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/clojure/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/clojure/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/clojure/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/clojure/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/clojure/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/clojure/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/clojure/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/clojure/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/clojure/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/common-lisp/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/common-lisp/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/common-lisp/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/common-lisp/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/common-lisp/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/common-lisp/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/common-lisp/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/common-lisp/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/common-lisp/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/common-lisp/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/css/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/css/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/css/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/css/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/css/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/css/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/css/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/css/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/css/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/css/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/cuda/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/cuda/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/cuda/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/cuda/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/cuda/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/cuda/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/cuda/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/cuda/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/cuda/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/cuda/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/dart/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/dart/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/dart/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/dart/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/dart/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/dart/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/dart/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/dart/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/dart/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/dart/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/erlang/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/erlang/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/erlang/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/erlang/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/erlang/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/erlang/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/erlang/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/erlang/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/erlang/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/erlang/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/fortran/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/fortran/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/fortran/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/fortran/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/fortran/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/fortran/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/fortran/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/fortran/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/fortran/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/fortran/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/go/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/go/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/go/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/go/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/go/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/go/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/go/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/go/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/go/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/go/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/haskell/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/haskell/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/haskell/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/haskell/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/haskell/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/haskell/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/haskell/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/haskell/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/haskell/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/haskell/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/html/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/html/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/html/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/html/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/html/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/html/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/html/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/html/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/html/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/html/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java-server-pages/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java-server-pages/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java-server-pages/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java-server-pages/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java-server-pages/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java-server-pages/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java-server-pages/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java-server-pages/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java-server-pages/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/java-server-pages/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/javascript/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/javascript/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/javascript/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/javascript/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/javascript/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/javascript/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/javascript/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/javascript/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/javascript/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/javascript/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/julia/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/julia/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/julia/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/julia/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/julia/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/julia/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/julia/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/julia/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/julia/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/julia/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/lua/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/lua/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/lua/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/lua/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/lua/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/lua/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/lua/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/lua/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/lua/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/lua/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/markdown/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/markdown/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/markdown/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/markdown/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/markdown/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/markdown/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/markdown/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/markdown/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/markdown/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/markdown/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/matlab/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/matlab/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/matlab/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/matlab/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/objective-c++/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/objective-c++/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/objective-c++/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/objective-c++/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/objective-c++/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/objective-c++/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/objective-c++/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/objective-c++/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/objective-c++/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/objective-c++/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ocaml/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ocaml/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ocaml/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ocaml/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ocaml/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ocaml/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ocaml/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ocaml/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ocaml/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ocaml/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/opencl/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/opencl/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/opencl/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/opencl/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/opencl/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/opencl/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/opencl/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/opencl/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/opencl/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/opencl/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/pascal/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/pascal/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/pascal/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/pascal/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/pascal/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/pascal/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/pascal/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/pascal/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/pascal/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/pascal/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/perl/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/perl/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/perl/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/perl/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/perl/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/perl/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/perl/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/perl/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/perl/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/perl/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/php/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/php/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/php/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/php/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/php/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/php/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/php/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/php/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/php/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/php/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/python/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/python/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/python/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/python/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/python/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/python/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/python/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/python/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/python/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/python/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/r/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/r/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/r/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/r/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/r/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/r/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/r/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/r/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/r/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/r/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/restructuredtext/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/restructuredtext/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/restructuredtext/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/restructuredtext/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/restructuredtext/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/restructuredtext/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/restructuredtext/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/restructuredtext/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/restructuredtext/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/restructuredtext/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rmarkdown/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rmarkdown/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rmarkdown/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rmarkdown/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rmarkdown/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rmarkdown/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rmarkdown/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rmarkdown/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rmarkdown/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rmarkdown/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ruby/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ruby/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ruby/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ruby/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ruby/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ruby/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ruby/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ruby/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ruby/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/ruby/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rust/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rust/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rust/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rust/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rust/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rust/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rust/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rust/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rust/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/rust/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/sql/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/sql/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/sql/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/sql/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/sql/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/sql/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/sql/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/sql/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/sql/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/sql/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/swift/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/swift/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/swift/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/swift/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/swift/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/swift/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/swift/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/swift/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/swift/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/swift/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/systemverilog/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/systemverilog/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/systemverilog/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/systemverilog/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/systemverilog/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/systemverilog/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/systemverilog/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/systemverilog/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/systemverilog/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/systemverilog/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/tcl/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/tcl/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/tcl/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/tcl/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/tcl/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/tcl/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/tcl/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/tcl/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/tcl/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/tcl/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/typescript/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/typescript/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/typescript/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/typescript/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/typescript/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/typescript/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/typescript/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/typescript/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/typescript/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/typescript/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vhdl/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vhdl/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vhdl/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vhdl/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vhdl/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vhdl/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vhdl/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vhdl/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vhdl/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vhdl/quality_p95/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vue/quality_p50/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vue/quality_p55/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vue/quality_p60/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vue/quality_p65/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vue/quality_p70/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vue/quality_p75/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vue/quality_p80/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vue/quality_p85/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vue/quality_p90/allenai/dolma2-tokenizer/*.npy",
    "s3://ai2-llm/preprocessed/bigcode_commitpack/dolma-3_5-languages_tagged_resharded_rewritten_partitioned/vue/quality_p95/allenai/dolma2-tokenizer/*.npy",
]


DATASET_CONFIG = SourceMixtureList(
    sources=[
        SourceMixtureConfig(
            source_name="all-dressed-snazzy2-v0.1-150b",
            target_ratio=0.5,
            paths=[f"{DOLMA_3_PREFIX}/{TOKENIZER_CONFIG.identifier}/all-dressed-snazzy2/**/*.npy"],
        ),
        SourceMixtureConfig(
            source_name="code-commitpack",
            target_ratio=0.5,
            # about 154B tokens
            paths=COMMIT_PACK_PATHS,
        ),
    ]
)


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )
    root_dir = get_root_dir(cli_context.cluster)
    work_dir = get_work_dir(root_dir)
    save_dir = f"{BASE_SAVE_DIR}/{get_beaker_username()}/{cli_context.run_name}"

    beaker_launch_config: BeakerLaunchConfig | None = build_launch_config(
        name=cli_context.run_name,
        cmd=cli_context.remote_cmd,
        cluster=cli_context.cluster,
        root_dir=root_dir,
        workspace=WORKSPACE,
        num_nodes=NUM_NODES,
        nccl_debug=True,
    )
    beaker_launch_config.priority = PRIORITY

    train_module_config: TransformerTrainModuleConfig = cookbook.configure_train_module(
        max_sequence_length=SEQ_LENGTH,
        rank_microbatch_size=SEQ_LENGTH * 2,
        learning_rate=LR,
        scheduler=LinearWithWarmup(units=SchedulerUnits.steps, warmup=200, alpha_f=0.0),
        activation_memory_budget=0.5,
    )

    DATASET_CONFIG.validate()
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=DATASET_CONFIG,
            requested_tokens=MAX_TOKENS,
            global_batch_size=GLOBAL_BATCH_SIZE,
            processes=16,
            seed=SEED,
        ),
        tokenizer=TOKENIZER_CONFIG,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=4
    )

    trainer_config = cookbook.configure_trainer(
        load_path=LOAD_PATH,
        load_trainer_state=False,
        load_optim_state=True,
        max_duration=Duration.tokens(MAX_TOKENS),
        checkpoint_dir=save_dir,
        work_dir=work_dir,
    ).with_callbacks(
        cookbook.configure_default_callbacks(
            run_name=run_name_with_ts,
            wandb_group_name=cli_context.run_name,
            checkpoint_save_interval=None,
            ephemeral_checkpoint_save_interval=250,
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=MODEL_CONFIG,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=SEED,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


def main():
    # TWO CASES:
    # 1. len(sys.argv) < 4: definitely i have to add the run name here
    # 2. len(sys.argv) >= 4: depends on whether the first 4 all start NOT
    #                        with --: if they do, then i already have the
    #                        run name. but if any of the first 4 start
    #                        with --, then it means that some are already
    #                        overrides, so i need to add the run name in.
    if len(sys.argv) < 4 or any(arg.startswith("--") for arg in sys.argv[:4]):
        sys.argv.insert(2, Path(__file__).stem)

    # now i can just call the main function
    return olmo_core_main(config_builder=build_experiment_config)


if __name__ == "__main__":
    """
    Invoke this script directly to access the internal experiment CLI, which
    supports launch, train, dry_run, and other subcommands.

    Examples:
        To render the config and exit:
            uv run src/.../50web_alldressed_v2_50stack_edu_python.py dry_run

        To launch a training run on Augusta w/ 8 nodes:
            uv run src/.../50web_alldressed_v2_50stack_edu_python.py launch ai2/augusta \
                --launch.num_nodes=8 \
                --launch.priority=high
    """
    main()
