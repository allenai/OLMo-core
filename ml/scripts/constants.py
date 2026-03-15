"""
"""

import os

DEFAULT_DIR_PATH = '/'.join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-3])


## Default hyperparameters for all models, and specific models. The "all" key applies to all models, and the specific model keys override the "all" key for that model.
## Defaults to chinchilla: D = 20N. Should be adjusted to D = 100N. Consider seq length 4096, global batch size 256.
MODEL_HP_DEFAULTS = {
    "all": {
        "global_batch_size": [512],
        "sequence_length": [2048],
        "train_module": {
            "optim": {
                "weight_decay": [0.1],
            },
            "scheduler": {
                "units": ["steps"],
                "warmup_steps": [2000],
            },
        },
        "trainer": {
            "max_duration": {
                "unit": ["tokens"],
            },
        },
    },
    "olmo2_10M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [1000000000],
            },
        },
    },
    "olmo2_20M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [2000000000],
            },
        },
    },
    "olmo2_50M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [5000000000],
            },
        },
    },
    "olmo2_100M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [10000000000],
            },
        },
    },
    "olmo2_200M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [20000000000],
            },
        },
    },
    "olmo2_400M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [40000000000],
            },
        },
    }
}

PROJECT_SPECS = {
    "margsli": {
        'DEFAULT_SAVE_PATH': os.path.join(DEFAULT_DIR_PATH, 'models'),
        'DATA_WORK_DIR': "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data",
        'VALID_DATA_DIR': "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/preprocessed",
        "WANDB_PROJECT": "moe",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "moe",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "zlab",
        "SLURM_PARTITION": "ckpt-g2",
        "COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
        "NUM_GPUS": 4,
        "MODEL": [],
        "DATAROOT": "https://olmo-data.org/",
        "DATA_DIR": "/gscratch/zlab/snehark/OLMo-core/data/",
        "NAME_KEYS": [],
    },
    "ubuntu": {
          'DEFAULT_SAVE_PATH': os.path.join(DEFAULT_DIR_PATH, 'models'),
          'DATA_WORK_DIR': os.path.join(DEFAULT_DIR_PATH, 'ml/data'),
          'VALID_DATA_DIR': os.path.join(DEFAULT_DIR_PATH, 'ml/data/preprocessed'),
          "WANDB_PROJECT": "moe",
          "WANDB_ENTITY": "ml-moe",
          "CONDA_ENV_NAME": "olmoe-core",
          "PROJECT_DIR": DEFAULT_DIR_PATH,
          "SLURM_ACCOUNT": "",
          "SLURM_PARTITION": "",
          "COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
          "NUM_GPUS": 1,
          "MODEL": [],
          "DATAROOT": "https://olmo-data.org/",
          "DATA_DIR": os.path.join(DEFAULT_DIR_PATH, 'data'),
          "NAME_KEYS": [],
      },
}

HARDWARE_SPECS_DICT = {
    "all": {
        "NUM_GPUS": 4,
        "NUM_CPUS": 5,
        "MEM_GB": 120,
        "per_gpu_batch_size": 16,
    },
    "ckpt-g2": {
        "JOBTIME": "9:00:00",
    },
    "olmo2_10M": { 
    },
    "olmo2_20M": { 
    },
    "olmo2_50M": { 
    },
    "olmo2_100M": { 
        "gpu-h200": {
            "per_gpu_batch_size": 16,
            "NUM_CPUS": 16,
            "MEM_GB": 240,
        }, 
    },
    "olmo2_200M": { 
        "gpu-l40": {
            "per_gpu_batch_size": 8,
        }, 
        "gpu-a40": {
            "per_gpu_batch_size": 8,
        }, 
        "gpu-h200": {
            "per_gpu_batch_size": 16,
            "NUM_CPUS": 16,
            "MEM_GB": 240,
        }, 
    },
    "olmo2_400M": { 
        "gpu-l40": {
            "per_gpu_batch_size": 8,
        }, 
        "gpu-a40": {
            "per_gpu_batch_size": 8,
        }, 
    }
}
