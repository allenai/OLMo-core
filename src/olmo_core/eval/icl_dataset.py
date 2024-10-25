from abc import ABCMeta
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Union

from ..data.tokenizer import Tokenizer


class ICLTaskDataset(metaclass=ABCMeta):
    metric_type: ClassVar[str]

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split: str = "validation",
        metric_type: Optional[str] = None,  # Override default metric type
        prompts: Optional[List[Optional[str]]] = None,  # List of prompt variants to use
    ):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.prompts = prompts or [None]
        self.current_prompt = None
        if metric_type is not None:
            self.metric_type = metric_type
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check

        self.samples: List[Dict[str, Any]] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        dataset_list = []
        for ds_name in dataset_names:
            dataset = load_hf_dataset(self.dataset_path, ds_name, split)
            dataset_list.append(dataset)
        self.dataset = datasets.concatenate_datasets(dataset_list)

        # prep examples
        self.prep_examples()
