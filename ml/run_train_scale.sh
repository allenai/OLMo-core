TRAIN_DATA_FILES="data/part-00000-e9c46804-8d86-45cf-8ebc-4744ff73914d-c000.parquet"
for i in $(seq -f "%05g" 1 8)
do
    TRAIN_DATA_FILES=${TRAIN_DATA_FILES},data/part-${i}-e9c46804-8d86-45cf-8ebc-4744ff73914d-c000.parquet
done

# export ROOT_DIR="$(dirname "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )")"
export ROOT_DIR=/mmfs1/gscratch/zlab/margsli/gitfiles/open_lm_scaling

export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

python $ROOT_DIR/