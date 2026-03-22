# convert checkpoint (change input/output in script)

# old ckpt
python /workspace/OLMo-core/src/olmo_core/nn/moe/v2/hf/convert_checkpoint.py --output-path /workspace/checkpoint/OLMoE3-dev-260304-decay-2000B-100B_2560d3072a_24L2560M2560S_48E2K1S_c3/step96085-hf --ckpt-path /workspace/checkpoint/OLMoE3-dev-260304-decay-2000B-100B_2560d3072a_24L2560M2560S_48E2K1S_c3/step96085

python /workspace/OLMo-core/src/olmo_core/nn/moe/v2/hf/convert_checkpoint.py --output-path /workspace/checkpoint/OLMoE3-dev-260304-decay-2000B-100B-top4_2560d3072a_24L2560M2560S_48E4K1S_c3/step96085-hf --ckpt-path /workspace/checkpoint/OLMoE3-dev-260304-decay-2000B-100B-top4_2560d3072a_24L2560M2560S_48E4K1S_c3/step96085

# peri ln + embed scale + embed norm
python /workspace/OLMo-core/src/olmo_core/nn/moe/v2/hf/convert_checkpoint.py --output-path /workspace/checkpoint/OLMoE3-dev-260319_2560d3072a_24L2560M1280S_48E4K1S_c1/step44500-hf --ckpt-path /workspace/checkpoint/OLMoE3-dev-260319_2560d3072a_24L2560M1280S_48E4K1S_c1/step44500

# test converted checkpoint (change path in script)
python test_olmo3moe.py
