"""Launch the sharedvec-dolci25-tokenmatch arm; see the shared builder for the comparison contract."""

from functools import partial

from _qwen3_sharedvec_33344_common import build_experiment_config

from olmo_core.internal.experiment import main

if __name__ == "__main__":
    main(config_builder=partial(build_experiment_config, arm="sharedvec-dolci25-tokenmatch"))
