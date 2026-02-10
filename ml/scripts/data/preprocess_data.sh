conda activate moe
cd /gscratch/zlab/margsli/gitfiles/olmoe

export DIR=/gscratch/zlab/margsli/gitfiles
export OLMOE_DIR=$DIR/olmoe-core/OLMo-core
export DATA_DIR=$OLMOE_DIR/ml/data
export RAW_DATA_DIR=$DATA_DIR/raw/OLMoE-mix-0924/data

for d in $RAW_DATA_DIR/wiki/ ; do
    mkdir -p ${d//raw/preprocessed} ;
    dolma tokens \
    --documents ${d}/* \
    --destination ${d//raw/preprocessed}/ \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes 20 ;
done