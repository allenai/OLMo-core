# convert checkpoint (change input/output in script)
python convert_checkpoint.py
python /workspace/OLMo-core/src/olmo_core/nn/moe/v2/hf/convert_checkpoint.py --output-path /workspace/checkpoint/OLMoE3-abl-260102-010d4_1024d1024a_12L768M768S_32E4K1S_abl/step33000-hf --ckpt-path /workspace/checkpoint/OLMoE3-abl-260102-010d4_1024d1024a_12L768M768S_32E4K1S_abl/step33000


# test converted checkpoint (change path in script)
python test_olmo3moe.py
