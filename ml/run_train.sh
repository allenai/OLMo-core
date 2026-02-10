HF_MODEL=Qwen/Qwen2-0.5B
HF_DATASET=rulins/mmlu_searched_results_from_massiveds

export ROOT_DIR="$(dirname "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )")"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

python $ROOT_DIR/