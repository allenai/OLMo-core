#!/bin/bash

uv run src/scripts/train/ladder/olmo3_ladder.py metrics-all --cluster ai2/jupiter --name "olmo3-baseline-ladder" --chinchilla-multiple 8.0 --local --output-dir /Users/tylerr/workspace/OLMo-core/scratch/laddermetrics
uv run src/scripts/train/ladder/olmo3_ladder.py metrics-all --cluster ai2/jupiter --name "olmo3-gated-attn" --chinchilla-multiple 8.0 --local --output-dir /Users/tylerr/workspace/OLMo-core/scratch/laddermetrics
uv run src/scripts/train/ladder/olmo3_ladder.py metrics-all --cluster ai2/jupiter --name "olmo3-gnope" --chinchilla-multiple 8.0 --local --output-dir /Users/tylerr/workspace/OLMo-core/scratch/laddermetrics
uv run src/scripts/train/ladder/olmo3_ladder.py metrics-all --cluster ai2/jupiter --name "olmo3-instance-packing" --chinchilla-multiple 8.0 --local --output-dir /Users/tylerr/workspace/OLMo-core/scratch/laddermetrics
uv run src/scripts/train/ladder/olmo3_ladder.py metrics-all --cluster ai2/jupiter --name "olmo3-cautious-wd" --chinchilla-multiple 8.0 --local --output-dir /Users/tylerr/workspace/OLMo-core/scratch/laddermetrics
uv run src/scripts/train/ladder/olmo3_ladder.py metrics-all --cluster ai2/jupiter --name "olmo3-hybrid-gdn" --chinchilla-multiple 8.0 --local --output-dir /Users/tylerr/workspace/OLMo-core/scratch/laddermetrics
uv run src/scripts/train/ladder/olmo3_ladder.py metrics-all --cluster ai2/jupiter --name "olmo3-hybrid-gdn-deux" --chinchilla-multiple 8.0 --local --output-dir /Users/tylerr/workspace/OLMo-core/scratch/laddermetrics
uv run src/scripts/train/ladder/olmo3_ladder.py metrics-all --cluster ai2/jupiter --name "olmo3-muon-2xBS" --chinchilla-multiple 8.0 --local --output-dir /Users/tylerr/workspace/OLMo-core/scratch/laddermetrics
uv run src/scripts/train/ladder/olmo3_ladder.py metrics-all --cluster ai2/jupiter --name "olmo3-muon" --chinchilla-multiple 8.0 --local --output-dir /Users/tylerr/workspace/OLMo-core/scratch/laddermetrics
uv run src/scripts/train/ladder/olmo3_ladder.py metrics-all --cluster ai2/jupiter --name "olmo3-peri-ln" --chinchilla-multiple 8.0 --local --output-dir /Users/tylerr/workspace/OLMo-core/scratch/laddermetrics