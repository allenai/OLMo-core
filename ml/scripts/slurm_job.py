"""Contains basic helper functions for running a parameter sweep on the Hyak
cluster using the SLURM scheduler.
Adapted from ParlAI
"""

from collections import namedtuple
from copy import deepcopy
import collections.abc
import hashlib
import json
import os
import random
import subprocess
import sys
from utils import dict_update, has_file_been_modified_recently

RANDOM_PORT_MIN = 10000
RANDOM_PORT_MAX = 20000

BASH_IF_CLAUSE = """
if [[ "$SLURM_ARRAY_TASK_ID" == "{index}" ]]; then
    bash {SAVE}/run.sh >> {SAVE}/stdout 2>> {SAVE}/stderr
fi
"""
SLRM_JOB_ARRAY_TEMPLATE = """
#!/bin/bash
#SBATCH --job-name={SWEEP_NAME}
#SBATCH --output={SAVE_ROOT}/slurm_logs/stdout.%j
#SBATCH --error={SAVE_ROOT}/slurm_logs/stderr.%j
#SBATCH --account={account}
#SBATCH --partition={partition}
## make sure we don't clobber log files if jobs get restarted
#SBATCH --open-mode=append
#SBATCH --nodes={nodes}
#SBATCH --time={jobtime}
## make sure we are told about preempts, and jobs running out of time, 5 min beforehand
#SBATCH --signal=USR1@60
## number of cpus *per task*. Highly recommend this to be 10.
#SBATCH --cpus-per-task={cpus}
## srun forks ntasks_per_node times on each node
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --mem={mem_gb}G
{SBATCH_EXTRAS}

source ~/.bashrc
{conda_command}

echo "# -------- BEGIN CALL TO run.sh --------"
# -K kills all subtasks if one particular task crashes. This is necessary for
# distributed training
{JOB_LAUNCHER}
"""

SH_TEMPLATE = """
#!/bin/bash
set -e

# stores the child process
CHILD=""

# handles a TERM signal
term_handler () {{
    # catch and ignore TERM. we get multiple terms during shutdown, so best
    # to just do nothing
    # but still keep going with the python process
    wait "$CHILD"
}}

# handles an interrupt (aka ctrl-C)
int_handler () {{
    # propagate a ctrl-C to the python process
    kill -s INT "$CHILD"
    wait "$CHILD"
}}

# handles a USR1, which signals preemption or job time is up
usr1_handler () {{
    echo "SLURM signaling preemption/times up (SLURM_PROCID $SLURM_PROCID)."
    kill -s INT "$CHILD"  # send ctrl-c to python
    if {SHOULD_REQUEUE} && [ "$SLURM_PROCID" -eq "0" ]; then
        echo "Waiting 5s and resubmitting..."
        sleep 5
        echo "Resubmitting..."
        scontrol requeue $SLURM_JOB_ID
    fi
    wait "$CHILD"
}}


trap 'int_handler' INT
trap 'usr1_handler' USR1
trap 'term_handler' TERM

# Uncommenting these two lines can help with identifying hangs
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# setting this this can also help with hangs
# NCCL_LL_THRESHOLD=0

# if in distributed, make sure we're using the actual network
export NCCL_SOCKET_IFNAME=^docker0,lo
echo
nvidia-smi

source ~/.bashrc

echo "SLURM_PROCID"=$SLURM_PROCID
echo "node-list: $SLURM_JOB_NODELIST"

export MASTER_PORT={job_port}

export WORLD_SIZE=$(($NUM_GPUS))
echo "MASTER_PORT"=$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export WORLD_SIZE=$SLURM_NTASKS
echo "WORLD_SIZE="$WORLD_SIZE
export RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export LOCAL_RANK=$SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))
# ******************************************************************************************

# zoom zoom - recommended from lightning, copied from open_lm
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
######################

# olmo-core specific
export OLMO_SHARED_FS=1

cd {NEW_DIR_PATH}
export PYTHONPATH={SAVE_ROOT}/{repo_name}:$PYTHONPATH
if [[ "$SLURM_PROCID" == "0" ]]; then 
    CUDA_LAUNCH_BLOCKING=1 torchrun --nproc-per-node=gpu --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT {cmd} 
fi
echo "# -------- FINISHED CALL TO SRUN --------"
echo
nvidia-smi

"""


def sha1(string):
    """Compute the sha1 hexdigest of the string."""
    return hashlib.sha1(string.encode('utf-8')).hexdigest()


def run_grid(
    grid,
    default_grid={},
    sweep_name="",
    specs={},
    name_keys=[],
    prefix=None,
    gpus=1,
    cpus=10,
    nodes=1,
    node_exclude=None,
    account='zlab',
    partition='gpu-rtx6k',
    DIR_PATH="",
    jobtime='01:59:59',
    include_job_id=False,
    hashname=False,
    saveroot='',
    logroot='',
    mem_gb=64,
    requeue=False,
    data_parallel=False,
    comment=None,
    copy_env=True,
    copy_dirs=[],
    max_num_jobs=None,
    num_copies=1,
    job_id_start=1,
    debug_mode=False,
    dry_mode=False,
    dependencies=[],
    repo_name="code",
    conda_env_name=None,
    include_jobs_indices=None,
    filter_succeeded=True,
    filter_running=True,
    sweep_port_start=None,
):
    """Generates full commands from a grid.

    Arguments:
    grid -- (dict) keys are hyperparam strings (e.g. --learningrate or -lr),
        values are lists of parameter options (e.g. [0.5, 0.05, 0.005]).
        You can tie options together in a limited fashion (e.g.
        '--opt': ['sgd -lr 0.5', 'adam -lr 0.005']), but we don't support
        nesting dicts/lists yet.
    name_keys -- (set) contains any params to always include in the model
        filename (e.g. {'-hs'} will make sure that the filename includes
        _hs=X_). By default, any key with more than one value will also be
        included in the model filename.
    sweep_name -- (str) name of the sweep
    prefix -- (str) base command to run
    hashname -- (bool) if True, uses a hash of the parameters as the
        folder. Sometimes necessary for long commands (default False).
    dataparallel -- (bool) set to True if running with nn.DataParallel
    comment -- you need to add a text comment to use priority partition
    copy_env -- (bool) if True, copies local directory components to the
        save root, and uses this to run the jobs
    copy_dirs -- (list) list of additional directories to copy
    max_num_jobs -- (int) maximum number of jobs
    num_copies -- (int) number of copies of each job to run
    job_id_start -- (int) starting job id for numbering jobs
    debug_mode -- (bool) if True, runs only one job for debugging
    dry_mode -- (bool) if True, does not actually run the jobs, just prints
        the commands that would be run
    dependencies -- (list) list of job ids that this job depends on
    repo_name -- (str) name of the repository to copy
    conda_env_name -- (str) name of the conda environment to activate
    include_jobs_indices -- (list) list of job indices to include in the sweep
    filter_succeeded -- (bool) if True, filters out jobs that have already
        succeeded (i.e. have a log file with "got exitcode: 0")
    sweep_port_start -- (int) starting port for the sweep, if None, a random
        port will be chosen for each job
    """
    
    def get_name_keys(dictionary, parents_key_list=[], name_keys=[], use_all_keys=False):
        items = []
        for key, value in dictionary.items():
            new_key_list = deepcopy(parents_key_list)
            new_key_list.append(key)
            if isinstance(value, collections.abc.Mapping):
                items.extend(get_name_keys(value, new_key_list))
            else:
                assert isinstance(value, list)
                if isinstance(value[0], collections.abc.Mapping):
                    for vd in value:
                        if isinstance(vd, collections.abc.Mapping):
                            items.extend(get_name_keys(vd, new_key_list))
                elif len(value) > 1 or use_all_keys:
                    items.append('.'.join(new_key_list))
        items = list(set(items + name_keys))  # remove duplicates
        return items

    def make_job_name(name_keys_list, args_dict, sweep_name='', subgrid_name=''):
        name_list = []
        if sweep_name:
            name_list.append(sweep_name)
        if subgrid_name:
            name_list.append(subgrid_name)
        for key in name_keys_list:
            value = args_dict.get(key, None)
            if value is None or isinstance(value, collections.abc.Mapping):
                continue
            short_key = key.replace("_", "").split('.')[-1] if '.' in key else key
            if type(value) == str:
                value = value.replace('_', '')
                if ' ' in value:
                    value = value.replace(' --', '_').replace(' -', '_')
                    value = value.replace(' ', '=')
            name_list.append('{}={}'.format(short_key, str(value)))
        return '_'.join(name_list)
    
    def unroll_args(d, prefix=''):
        """Unrolls a dict of args into a list of strings."""
        args = {}
        for k, v in d.items():
            if isinstance(v, collections.abc.Mapping):
                args = dict_update(args, unroll_args(v, f'{prefix}.{k}' if prefix else k))
            else:
                if prefix:
                    args[f"{prefix}.{k}"] = v
                else:
                    args[k] = v
        return args
    

    def check_if_job_succeeded_before(job_name, save_root):
        """Check if a job has already been run before."""
        wandb_log_path = os.path.join(save_root, job_name, 'wandb', 'wandb', 'latest-run', 'logs', 'debug.log')
        if os.path.exists(wandb_log_path):
            with open(wandb_log_path, 'r') as f:
                s = f.read()
                if "got exitcode: 0" in s:
                    print(f"Job {job_name} already done before, skipping.")
                    return True
        return False
    
    # if updated in the last 10 minutes, assume it's running
    def check_if_job_is_running(job_name, save_root, recent_threshold_seconds=3600):
        """Check if a job is currently running."""
        stdout_path = os.path.join(save_root, job_name, 'stdout')
        running = has_file_been_modified_recently(stdout_path, recent_threshold_seconds=recent_threshold_seconds) 
        if running:
            print(f"Job {job_name} may be running right now, skipping.")
        return running

    

    if not prefix:
        raise ValueError('Need prefix command')
    SAVE_ROOT = saveroot
    LOG_ROOT = logroot

    Job = namedtuple('Job', ['cmd', 'name'])
    all_jobs = []
    name_key_lists = {}

    import itertools
    def c_prod(d):
        if isinstance(d, list):
            for i in d:
                yield from ([i] if not isinstance(i, (dict, list)) else c_prod(i))
        else:
            for i in itertools.product(*map(c_prod, d.values())):
                yield dict(zip(d.keys(), i))

    all_permutation_dicts = {}
    main_grid = dict_update(deepcopy(default_grid), grid["main_grid"])
    for subgrid_name, subgrid in grid["subgrids"].items():
        subgrid_merged = dict_update(deepcopy(main_grid), subgrid)
        # print(subgrid_merged)
        all_permutation_dicts[subgrid_name] = list(c_prod(subgrid_merged))
        name_key_lists[subgrid_name] = get_name_keys(subgrid_merged, name_keys=name_keys)

    # shorten names if possible
    if hashname:
        # keep the names from getting too long
        full_names = [name for _, _, name in all_jobs]
        cutoff = i = 4
        while i < 40:
            if len(set([n[1:i] for n in full_names])) == len(full_names):
                cutoff = i
                break
            i += 1
    else:
        cutoff = None

    final_jobs = []
    job_id = job_id_start

    for subgrid_name, permutations_dicts in all_permutation_dicts.items():
        name_key_list = name_key_lists[subgrid_name]
        for config_dict in permutations_dicts:
            for _ in range(num_copies):
                cmd_args = unroll_args(config_dict)
                name = make_job_name(name_key_list, cmd_args, sweep_name=sweep_name, subgrid_name=subgrid_name)
                name = name[:cutoff] if cutoff else name
                name = sha1(name) if hashname else name
                cmd = f"{prefix} {name} " + ' '.join([f'--{k}={v}' for k, v in cmd_args.items()])
                if include_job_id:
                    name += '/_jobid=' + str(job_id)
                final_jobs.append(Job(cmd=cmd, name=name))
                job_id += 1

    # Copy the directory if needed
    to_copy = [] + copy_dirs
    if copy_env and to_copy:
        bash('mkdir -p ' + os.path.join(SAVE_ROOT, repo_name))
        for c in to_copy:
            c_head, _ = os.path.split(c)
            # if subfolder, copy folder then subfolder
            if len(c_head) > 1:
                bash('mkdir {SAVE_ROOT}/{repo_name}/{c_head}'.format(**locals()))
            bash('cp -r {DIR_PATH}/{c} {SAVE_ROOT}/{repo_name}/{c}'.format(**locals()))
        NEW_DIR_PATH = '{SAVE_ROOT}/{repo_name}'.format(**locals())
    else:
        NEW_DIR_PATH = DIR_PATH


    # Filter out jobs based on debug mode, indices, and status
    if debug_mode and len(final_jobs) > 1:
        final_jobs = final_jobs[:1]
    elif include_jobs_indices:
        final_jobs = [final_jobs[i] for i in include_jobs_indices]
    if filter_succeeded:
        final_jobs = [job for job in final_jobs if not check_if_job_succeeded_before(job.name, SAVE_ROOT)]
    if filter_running:
        final_jobs = [job for job in final_jobs if not check_if_job_is_running(job.name, SAVE_ROOT)]

    
    print(f'Found a total of {len(final_jobs)} jobs. \nExample of first job:\n{final_jobs[0].cmd}\n')
    print(final_jobs)

    if dry_mode:
        return

    print(f'Launching! Your jobs will run for {jobtime}.')

    # Dump grid, specs, jobs to files
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)
        with open(os.path.join(SAVE_ROOT, 'grid.json'), 'w') as f:
            json.dump(grid, f)
        with open(os.path.join(SAVE_ROOT, 'specs.json'), 'w') as f:
            json.dump(specs, f)
        with open(os.path.join(SAVE_ROOT, 'jobs_lookup.jsonl'), 'w') as f:
            for i, job in enumerate(final_jobs):
                f.write(json.dumps({'i': i, 'name': job.name, 'cmd': job.cmd}) + '\n')
    
    jobs_path = []
    sweep_port_start = sweep_port_start or random.randint(RANDOM_PORT_MIN, RANDOM_PORT_MAX)
    for i, job in enumerate(final_jobs):
        jobs_path.append(
            create_job_files(
                sweep_name,
                SAVE_ROOT,
                LOG_ROOT,
                job.name,
                job.cmd,
                gpus=gpus,
                nodes=nodes,
                data_parallel=data_parallel,
                requeue=requeue,
                NEW_DIR_PATH=NEW_DIR_PATH,
                repo_name=repo_name,
                job_port=sweep_port_start+i,
            )
        )
    submit_array_jobs(
        SWEEP_NAME=sweep_name,
        SAVE_ROOT=SAVE_ROOT,
        gpus=gpus,
        cpus=cpus,
        nodes=nodes,
        node_exclude=node_exclude,
        account=account,
        partition=partition,
        jobtime=jobtime,
        DIR_PATH=DIR_PATH,
        mem_gb=mem_gb,
        requeue=requeue,
        data_parallel=data_parallel,
        comment=comment,
        NEW_DIR_PATH=NEW_DIR_PATH,
        jobs_path=jobs_path,
        dependencies=dependencies,
        conda_env_name=conda_env_name,
    )


def bash(bashCommand):
    result = subprocess.run(bashCommand.split(), capture_output=True, text=True)
    return result.stdout.strip()

def create_job_files(
    SWEEP_NAME,
    SAVE_ROOT,
    LOG_ROOT,
    job_name,
    cmd,
    job_args=[],
    gpus=1,
    nodes=1,
    data_parallel=False,
    requeue=False,
    NEW_DIR_PATH="",
    repo_name="",
    job_port=None,
):
    """Creates job folders and scripts"""
    
    SHOULD_REQUEUE = str(requeue).lower()
    SAVE = os.path.join(SAVE_ROOT, job_name)
    bash('mkdir -p ' + SAVE)
    LOG = os.path.join(LOG_ROOT, job_name)
    bash('mkdir -p ' + LOG)
    SCRIPTFILE = os.path.join(SAVE, 'run.sh')
    ARGS_STR = ' '.join(job_args)
    job_port = job_port or random.randint(RANDOM_PORT_MIN, RANDOM_PORT_MAX)

    if data_parallel or not gpus:
        ntasks_per_node = 1
    else:
        if gpus > 8:
            ntasks_per_node = 8
        else:
            ntasks_per_node = gpus
    with open(SCRIPTFILE, 'w') as fw:
        fw.write(SH_TEMPLATE.format(**locals()).lstrip())
    return SAVE


def submit_array_jobs(
    SWEEP_NAME,
    SAVE_ROOT,
    gpus=1,
    cpus=1,
    nodes=1,
    node_exclude=None,
    account='zlab',
    partition='gpu-rtx6k',
    jobtime='23:59:59',
    DIR_PATH="",
    mem_gb=64,
    requeue=False,
    data_parallel=False,
    comment=None,
    NEW_DIR_PATH="",
    jobs_path=[],
    dependencies=[],
    conda_env_name=None,
    append_to_sbatch_str=None,
):  
    """Submits the jobs as a SLURM job array."""
    if not jobs_path:
        raise ValueError("No jobs to submit.")

    i = 0
    SLURMFILE = os.path.join(SAVE_ROOT, f'run_{i}.slrm')
    while os.path.exists(SLURMFILE):
        i += 1
        SLURMFILE = os.path.join(SAVE_ROOT, f'run_{i}.slrm')
    if data_parallel or not gpus:
        ntasks_per_node = 1
    else:
        if gpus > 8:
            ntasks_per_node = 8
        else:
            ntasks_per_node = gpus
    SBATCH_EXTRAS = []
    if node_exclude is not None:
        # If any nodes are down, exclude them here
        SBATCH_EXTRAS.append('#SBATCH --exclude ' + str(node_exclude))

    constraints = []

    total_num_jobs = len(jobs_path) - 1

    # Request the number of GPUs (defaults to 1)
    if gpus > 0:
        if gpus > 8:
            gpustr = '#SBATCH --gpus-per-node=8'
        else:
            gpustr = '#SBATCH --gpus-per-node={}'.format(gpus)
        SBATCH_EXTRAS.append(gpustr)

    if constraints:
        SBATCH_EXTRAS.append("#SBATCH -C '{}'".format('&'.join(constraints)))
    
    
    if comment:
        SBATCH_EXTRAS.append('#SBATCH --comment="{}"'.format(comment))

    if dependencies:
        SBATCH_EXTRAS.append('#SBATCH --dependency="{}"'.format(','.join(['afterok:' + str(d) for d in dependencies])))

    conda_command = f'conda activate {conda_env_name}' if conda_env_name else ''

    # make sure sbatch extras are a string
    SBATCH_EXTRAS = "\n".join(SBATCH_EXTRAS)
    JOB_LAUNCHER = []
    for idx, each_path in enumerate(jobs_path):
        JOB_LAUNCHER.append(BASH_IF_CLAUSE.format(index=idx, SAVE=each_path, nodes=nodes))
    JOB_LAUNCHER = "\n".join(JOB_LAUNCHER)
    bash('mkdir -p ' + os.path.join(SAVE_ROOT, 'slurm_logs'))
    with open(SLURMFILE, 'w') as fw:
        fw.write(SLRM_JOB_ARRAY_TEMPLATE.format(**locals()).lstrip())
        
    print(bash('sbatch --array=0-{} {}'.format(total_num_jobs, SLURMFILE)))