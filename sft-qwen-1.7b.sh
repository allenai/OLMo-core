RUN_NAME=qwen3-1.7b-sft-10k
BASE_CKPT=/weka/oe-adapt-default/jacobm/repos/cse-579/checkpoints/Qwen3-1.7B-Base-oc/model_and_optim
CLUSTER=ai2/saturn
DATASET_PATH=/weka/oe-adapt-default/jacobm/repos/cse-579/datasets/Dolci-Think-SFT-32B-qwen3-olmo-thinker-10k

python src/scripts/train/sft/Qwen3-1.7B-SFT.py launch \
    ${RUN_NAME} \
    ${BASE_CKPT} \
    ${CLUSTER} \
    --dataset_path=${DATASET_PATH} \
    --budget=ai2/oe-adapt \
    --workspace=ai2/flex2 \
    --num_nodes=1 \
    --launch.priority=urgent \
    --launch.preemptible=true