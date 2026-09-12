"""Continue joint alignment with mixed text/vision loss or text-only supervision."""

from olmo_core.internal.experiment import main
from olmo_core.internal.vision_midtraining import build_config

if __name__ == "__main__":
    main(config_builder=build_config)
