```
ENV_NAME=olmoe-core
DATA_FOLDER=/gscratch/zlab/snehark/OLMo-core/data
OLMO_CORE_PATH=/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/

module load cuda/12.6.3
mamba create -n $ENV_NAME python=3.11
mamba activate $ENV_NAME

git clone https://github.com/sneha-rk/OLMo-core.git
cd OLMo-core
pip install -e .[all]

git clone --recursive https://github.com/sneha-rk/grouped_gemm
cd group_gemm
GROUPED_GEMM_CUTLASS=1 pip install .

RUN_NAME=test_name
srun -p ckpt-all --time=12:00:00 --nodes=1 --cpus-per-task=4 --mem=16G --gres=gpu:l40s:2 --pty /bin/bash
torchrun ${OLMO_CORE_PATH}/src/scripts/official/OLMo-2-smallmoe-train.py $RUN_NAME
```