import argparse
import json
import os
from datetime import datetime
from copy import copy
from slurm_job import run_grid
from constants import PROJECT_SPECS, HARDWARE_SPECS_DICT, MODEL_HP_DEFAULTS
from utils import dict_update


SWEEP_NAME_DEFAULT = 'data_rep_AC'
project = 'moe'
MODELS = [
    # 'olmo2_200M',
    # 'olmo2_100M',
    # 'olmo2_50M',
    # 'olmo2_20M',
    'olmo2_10M',
]

def main(
    sweep_name=SWEEP_NAME_DEFAULT,
    relaunch_path=None,
    relaunch_name=None,
    add_time_to_name='front',
    add_model_to_name='end',
    debug=False, 
    dry_mode=False,
    account=None, 
    partition=None,
    job_time='24:00:00',
    gpus=None,
    cpus=None,
    mem=None,
    include_jobs_indices=None,
    ignore_specs_check_keys=["NUM_CPUS", "MEM_GB", "JOBTIME"],
    filter_succeeded=True,
    filter_running=True,
    **kwargs,
):
    if account is None or partition is None:
        raise RuntimeError("Must specify account and partition")

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    job_time = '1:00:00' if debug else job_time
    user = os.environ.get('USER')
    if user not in PROJECT_SPECS:
        raise ValueError(f"User {user} not found in PROJECT_SPECS. Please add your user to the PROJECT_SPECS dictionary.")
    USER_SPECS = PROJECT_SPECS[user]

    if relaunch_path or relaunch_name:
        if relaunch_name and relaunch_path:
            raise ValueError("Cannot specify both relaunch_name and relaunch_path")
        if relaunch_name:
            relaunch_path = os.path.join(PROJECT_SPECS[user]['DEFAULT_SAVE_PATH'], relaunch_name)
        relaunch_path = relaunch_path.rstrip('/')
        model_sweep_name = os.path.basename(relaunch_path)
        path_to_grid_file = os.path.join(relaunch_path, 'grid.json')
        path_to_specs = os.path.join(relaunch_path, 'specs.json')
        if not os.path.exists(path_to_grid_file):
            raise FileNotFoundError(f"Grid file {path_to_grid_file} does not exist.")
        grid = json.load(open(path_to_grid_file, 'r'))
        model = grid.get('main_grid', {}).get('model_name', [None])[0]

        SPECS = copy(USER_SPECS)
        SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT.get('all', {}))
        SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT.get(partition, {}))
        SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT[model].get(partition, {})) 
        SPECS['NUM_GPUS'] = gpus or SPECS['NUM_GPUS']
        SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
        SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]

        if os.path.exists(path_to_specs):
            old_specs = json.load(open(path_to_specs, 'r'))
            for key in old_specs:
                if key not in ignore_specs_check_keys:
                    assert SPECS.get(key) == old_specs[key], f"Specs mismatch for {key}: {SPECS.get(key)} != {old_specs[key]}"
        
        run_grid(
            grid,
            default_grid=dict_update(copy(MODEL_HP_DEFAULTS['all']), MODEL_HP_DEFAULTS.get(model, {})),
            sweep_name=model_sweep_name,
            specs=SPECS,
            name_keys=SPECS.get("NAME_KEYS", []),
            prefix=SPECS['COMMAND_PREFIX'],
            gpus=SPECS['NUM_GPUS'],
            cpus=SPECS["NUM_CPUS"],
            nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
            node_exclude=None,
            account=account,
            partition=partition,
            DIR_PATH=SPECS["PROJECT_DIR"],
            jobtime=(job_time if job_time else SPECS.get("JOBTIME", '24:00:00')),            
            include_job_id=False,
            hashname=False,
            saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
            logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
            mem_gb=SPECS["MEM_GB"],
            requeue=True,
            data_parallel=False,
            comment=None,
            copy_env=True,
            copy_dirs=[],
            max_num_jobs=None,
            num_copies=1,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            dependencies=[],
            repo_name="olmoe-core",
            conda_env_name=SPECS.get("CONDA_ENV_NAME"),
            include_jobs_indices=include_jobs_indices,
            filter_succeeded=filter_succeeded,
            filter_running=filter_running,
            # append_to_sbatch_str=None,
        )
        
    else:
        SWEEP_NAME = sweep_name
        if add_time_to_name == 'front':
            time_str = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
            SWEEP_NAME = f"{time_str}_{SWEEP_NAME}" if SWEEP_NAME else time_str
        for model in MODELS:
            model_sweep_name = f"{SWEEP_NAME}_{model}" if add_model_to_name == 'end' else SWEEP_NAME

            SPECS = copy(USER_SPECS)
            SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT.get('all', {}))
            SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT.get(partition, {}))
            SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT[model].get(partition, {}))
            SPECS['NUM_GPUS'] = gpus or SPECS['NUM_GPUS']
            SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
            SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]
            grid = {
                # main_grid is the top-level grid, the sweep will run over all combinations of these hyperparameters, 
                # combined with the subgrids
                "main_grid": { 
                    "model_name": [model],
                    "save_root": [f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}"],
                    # "scheduler": ["wsd"],
                    "moe_type": ["dropless"],
                    # "moe_bias_gamma": [0.001],  # None for default, or specify a float value
                    # "moe_lb_loss_weight": [0.0001],  # Weight for the lb-loss in MoE
                    'train_module': {
                        'optim': {
                            # 'lr': [4e-3, 1e-2],
                            'lr': [4e-4],
                        },
                    },
                    "trainer": {
                        "max_duration": {
                            # "value": [2000000000], #just testing
                        },
                    },
                },
                # allows you to bundle multiple hyperparameters together
                "subgrids": {
                    # "e1x1c1": {"moe_num_experts_list": ["1"]},
                    ### no generalist models
                    # "e2x1c1_nogen": {"moe_num_experts_list": ["2"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4x1c1_nogen": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8x1c1_nogen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x1c1_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4x0.5c2_nogen": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8x0.5c2_nogen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x0.5c2_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8x0.25c4_nogen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x0.25c4_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e16x0.125c8_nogen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4,8x0.5,0.25c1,2_nogen": {"moe_num_experts_list": ["4,8"], "moe_hidden_multipliers_list": ["0.5,0.25"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8,16x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e4,16x0.5,0.125c1,4_nogen": {"moe_num_experts_list": ["4,16"], "moe_hidden_multipliers_list": ["0.5,0.125"], "moe_router_top_ks_list": ["1,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e32x0.25c4_nogen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e64x0.25c4_nogen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e32x0.125c8_nogen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e64x0.125c8_nogen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"], "moe_generalist_hidden_multiplier": ["0"]},
                    # "e8,32x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["8,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16,32x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e32,32x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["32,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    "e32,64x0.25,0.125c2,4_nogen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"], "moe_generalist_hidden_multiplier": ["0"]},
                    ## 0.5 generalist models
                    # "e4x0.5c1_0.5gen": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8x0.5c1_0.5gen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16x0.5c1_0.5gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["1"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8x0.25c2_0.5gen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16x0.25c2_0.5gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16x0.125c4_0.5gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32x0.25c2_0.5gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e64x0.25c2_0.5gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32x0.125c4_0.5gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e64x0.125c4_0.5gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8,32x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["8,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e16,32x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32,32x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["32,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32,64x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8,16x0.25,0.125c1,2_0.5gen": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e8x0.25c3_0.25gen": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e16x0.25c3_0.25gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e16x0.125c6_0.25gen": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e4,16x0.5,0.125c1,2_0.25gen": {"moe_num_experts_list": ["4,16"], "moe_hidden_multipliers_list": ["0.5,0.125"], "moe_router_top_ks_list": ["1,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e8,16x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},"e32x0.125c4_0.5gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0.5"]},
                    # "e32x0.25c3_0.25gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e64x0.25c3_0.25gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["3"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e32x0.125c6_0.25gen": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e64x0.125c6_0.25gen": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["6"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e8,32x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["8,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e16,32x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["16,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e32,32x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["32,32"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    # "e32,64x0.25,0.125c2,2_0.25gen": {"moe_num_experts_list": ["32,64"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,2"], "moe_generalist_hidden_multiplier": ["0.25"]},
                    ## === Data repetition experiments (A+C) ===
                    ## Dense baselines at different repetition levels
                    "dense_rep1x": {"moe_num_experts_list": ["1"], "unique_data_fraction": ["1.0"], "num_repetitions": ["1"]},
                    "dense_rep2x": {"moe_num_experts_list": ["1"], "unique_data_fraction": ["0.5"], "num_repetitions": ["2"]},
                    "dense_rep4x": {"moe_num_experts_list": ["1"], "unique_data_fraction": ["0.25"], "num_repetitions": ["4"]},
                    "dense_rep8x": {"moe_num_experts_list": ["1"], "unique_data_fraction": ["0.125"], "num_repetitions": ["8"]},
                    ## MoE 32 experts at different repetition levels
                    "moe32_rep1x": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["1.0"], "num_repetitions": ["1"]},
                    "moe32_rep2x": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.5"], "num_repetitions": ["2"]},
                    "moe32_rep4x": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.25"], "num_repetitions": ["4"]},
                    "moe32_rep8x": {"moe_num_experts_list": ["32"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.125"], "num_repetitions": ["8"]},
                    ## MoE 64 experts at different repetition levels
                    "moe64_rep1x": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["1.0"], "num_repetitions": ["1"]},
                    "moe64_rep2x": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.5"], "num_repetitions": ["2"]},
                    "moe64_rep4x": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.25"], "num_repetitions": ["4"]},
                    "moe64_rep8x": {"moe_num_experts_list": ["64"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"], "moe_generalist_hidden_multiplier": ["0"], "unique_data_fraction": ["0.125"], "num_repetitions": ["8"]},
                },
            }

            run_grid(
                grid,
                default_grid=dict_update(copy(MODEL_HP_DEFAULTS['all']), MODEL_HP_DEFAULTS.get(model, {})),
                sweep_name=model_sweep_name,
                specs=SPECS,
                name_keys=SPECS.get("NAME_KEYS", []),
                prefix=SPECS['COMMAND_PREFIX'],
                gpus=SPECS['NUM_GPUS'],
                cpus=SPECS["NUM_CPUS"],
                nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
                node_exclude=None,
                account=account,
                partition=partition,
                DIR_PATH=SPECS["PROJECT_DIR"],
                jobtime=(job_time if job_time else SPECS.get("JOBTIME", '24:00:00')),      
                include_job_id=False,
                hashname=False,
                saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
                logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
                mem_gb=SPECS["MEM_GB"],
                requeue=True,
                data_parallel=False,
                comment=None,
                copy_env=True,
                copy_dirs=[],
                max_num_jobs=None,
                num_copies=1,
                job_id_start=1,
                debug_mode=DEBUG_MODE,
                dry_mode=DRY_MODE,
                dependencies=[],
                repo_name="olmoe-core",
                conda_env_name=SPECS.get("CONDA_ENV_NAME"),
                include_jobs_indices=include_jobs_indices,
                filter_succeeded=filter_succeeded,
                filter_running=filter_running,
                # append_to_sbatch_str=None,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--sweep-name', type=str, default=SWEEP_NAME_DEFAULT)
    parser.add_argument('-rp', '--relaunch-path', type=str, default=None, help="Path to the sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.")
    parser.add_argument('-rn', '--relaunch-name', type=str, default=None, help="Name of sweep, also base of sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.")
    parser.add_argument('--add-time-to-name', type=str, default='front', choices=['front', 'none'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('-a', '--slurm-account', type=str)
    parser.add_argument('-p', '--slurm-partition', type=str)
    parser.add_argument('-t', '--job-time', type=str)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--cpus', type=int)
    parser.add_argument('--mem', type=str)
    parser.add_argument('-i', '--include-jobs-indices', type=str, default=None)
    parser.add_argument('-nf', '--no-filter', action='store_true', help="If set, will not filter out jobs that have already been run in the sweep. Useful for debugging.")

    args = parser.parse_args()

    main(
        sweep_name=args.sweep_name,
        relaunch_path=args.relaunch_path,
        relaunch_name=args.relaunch_name,
        add_time_to_name=args.add_time_to_name,
        debug=args.debug, 
        dry_mode=args.dry_mode,
        account=args.slurm_account, 
        partition=args.slurm_partition,
        job_time=args.job_time,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        include_jobs_indices=([int(i) for i in args.include_jobs_indices.split(",")] if args.include_jobs_indices else None),
        filter_running=not args.no_filter,
        filter_succeeded=not args.no_filter,
    )
