"""
Attention sink analysis using hook-based monitoring (inspired by GAP monitor).
This approach uses forward hooks instead of modifying the attention backend directly.
"""

import argparse
import functools as ft
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoTokenizer

from olmo_core.config import DType
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.generate.generation_module.transformer import TransformerGenerationModule
from olmo_core.nn.attention.backend import AttentionBackend, AttentionBackendName


@dataclass
class AttentionSinkMonitor:
    """
    Monitor for analyzing attention sink patterns using forward hooks.

    Registers hooks on attention backend modules to capture q, k tensors
    and compute attention statistics without modifying the backend code.
    """

    # Configuration
    token_idx: int = 4000  # Token position to analyze
    sink_range: Tuple[int, int] = (0, 100)  # Range of sink token indices
    local_range: Tuple[int, int] = (3900, 4000)  # Range of local token indices

    # Internal state
    _handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    _results: List[dict] = field(default_factory=list)
    _current_layer: int = field(default=0)

    def attach(self, model: nn.Module):
        """Attach hooks to all attention backend modules in the model."""
        # First detach any existing hooks
        self.detach()
        self._results = []
        self._current_layer = 0

        for name, module in model.named_modules():
            if isinstance(module, AttentionBackend):
                h = module.register_forward_pre_hook(
                    ft.partial(self._attention_hook, module_name=name)
                )
                self._handles.append(h)

    def detach(self):
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def reset(self):
        """Reset results for a new inference run (keeps hooks attached)."""
        self._results = []
        self._current_layer = 0

    @torch.no_grad()
    def _attention_hook(self, module: nn.Module, args, module_name: str):
        """
        Forward pre-hook that captures q, k tensors and computes attention statistics.

        The backend's forward method receives:
          qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        where q, k, v have shape (batch_size, seq_len, n_heads, head_dim)
        """
        qkv = args[0]

        # Handle both packed and unpacked QKV
        if isinstance(qkv, tuple):
            q, k, _v = qkv
        else:
            # Packed QKV - skip for now as it requires different handling
            return

        seq_len = q.shape[1]

        # Only analyze if sequence is long enough
        if seq_len <= self.token_idx:
            return

        # Transpose to (batch_size, n_heads, seq_len, head_dim) for attention computation
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)

        n_heads = q_t.shape[1]
        head_dim = q_t.shape[-1]

        # Handle GQA by repeating k to match q's head count
        n_kv_heads = k_t.shape[1]
        if n_kv_heads != n_heads:
            n_rep = n_heads // n_kv_heads
            k_t = k_t.repeat_interleave(n_rep, dim=1)

        # Get the query for the target token position
        t = self.token_idx
        k_before = k_t[:, :, :t, :]  # All keys before token t
        this_q = q_t[:, :, t, :].unsqueeze(-2)  # Query at token t: [B, H, 1, D]

        # Compute attention scores
        scores = torch.einsum('b h q d, b h k d -> b h q k', this_q, k_before) / (head_dim ** 0.5)

        # Determine the effective attention window start
        # window_size is (left, right) where (-1, -1) means no sliding window
        # For causal attention, right is typically 0, left is the window size - 1
        window_size = getattr(module, 'window_size', (-1, -1))
        window_start = 0  # Default: can attend from position 0
        if window_size != (-1, -1):
            window_left = window_size[0]
            if window_left >= 0:
                # Token at position t can only attend to [t - window_left - 1, t]
                window_start = max(0, t - window_left - 1)
                # Mask out tokens outside the sliding window
                key_positions = torch.arange(t, device=scores.device)
                valid_mask = key_positions >= window_start
                # Apply mask: set invalid positions to -inf before softmax
                scores = scores.masked_fill(~valid_mask.view(1, 1, 1, -1), float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # [B, H, 1, t]

        # Compute attention weights for sink and local tokens
        # Sink = first 100 tokens the model CAN attend to (accounting for sliding window)
        sink_size = self.sink_range[1] - self.sink_range[0]  # Typically 100
        effective_sink_start = window_start
        effective_sink_end = min(window_start + sink_size, t)

        # Local = last 100 tokens before token_idx (unchanged)
        local_idx = torch.arange(self.local_range[0], min(self.local_range[1], t), device=attn.device)
        sink_idx = torch.arange(effective_sink_start, effective_sink_end, device=attn.device)

        if len(sink_idx) > 0:
            sink_weight = attn[..., sink_idx].sum(dim=-1).sum() / n_heads
            percent_sink = float(sink_weight * 100)
        else:
            percent_sink = 0.0

        if len(local_idx) > 0:
            local_weight = attn[..., local_idx].sum(dim=-1).sum() / n_heads
            percent_local = float(local_weight * 100)
        else:
            percent_local = 0.0

        # A "sink layer" is one with full attention (no sliding window)
        is_sink_layer = 1 if window_size == (-1, -1) else 0

        self._results.append({
            'layer': self._current_layer,
            'module': module_name,
            'percent_sink': percent_sink,
            'percent_local': percent_local,
            'is_sink_layer': is_sink_layer,
        })

        self._current_layer += 1

    def get_results(self) -> List[dict]:
        """Return collected results."""
        return self._results

    def print_summary(self):
        """Print a summary of the attention sink analysis."""
        if not self._results:
            print("No results collected.")
            return

        avg_sink = sum(r['percent_sink'] for r in self._results) / len(self._results)
        avg_local = sum(r['percent_local'] for r in self._results) / len(self._results)

        print(f"\n{'='*60}")
        print(f"ATTENTION SINK SUMMARY")
        print(f"  Layers analyzed: {len(self._results)}")
        print(f"  Token position: {self.token_idx}")
        print(f"  Sink tokens: {self.sink_range[0]}-{self.sink_range[1]}")
        print(f"  Local tokens: {self.local_range[0]}-{self.local_range[1]}")
        print(f"  Avg % attention to sink: {avg_sink:.2f}%")
        print(f"  Avg % attention to local: {avg_local:.2f}%")
        print(f"{'='*60}")


class OlmoCoreModel:
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str = "allenai/dolma2-tokenizer",
        max_length: int = 10_000,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"

        self.stop_token_ids = []

        self.generation_config = GenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=max_length,
            max_new_tokens=1,
            temperature=1.0,
            top_p=1.0,
            do_sample=False,
            use_cache=True,
            stop_token_ids=self.stop_token_ids,
        )

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.generation_module = TransformerGenerationModule.from_checkpoint(
            checkpoint_dir=model_name,
            generation_config=self.generation_config,
            device=self.device,
            dtype=DType.bfloat16,
            attention_backend=AttentionBackendName.torch,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze attention sink patterns in transformer models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "/weka/oe-training-default/ai2-llm/checkpoints/amandab/olmo29_7b_140B-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-f1a781d3/step2385",
            "/weka/oe-training-default/ai2-llm/checkpoints/amandab/olmo28_140B_lc_64k-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-988d396f/step2385/",
            "/weka/oe-training-default/ai2-llm/checkpoints/amandab/olmo25_140B_lc_64k-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-b9609b3f/step2385",
            "/weka/oe-training-default/ai2-llm/checkpoints/amandab/llamalike_140B_lc_64k-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-1623f603/step2385/",
            "/weka/oe-training-default/ai2-llm/checkpoints/amandab/llamalike_140B_lc_64k-midtrain_round3_qwenlike_s2pdf_gzip2080_10B-preannealv2-ffc378a3/step2385/",
        ],
        help="List of paths to model checkpoints",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="long_docs",
        help="Path to the long_docs dataset (HuggingFace dataset format)",
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=None,
        help="Number of documents to process (default: all documents in dataset)",
    )
    parser.add_argument(
        "--token_idx",
        type=int,
        default=4000,
        help="Token index to compute attention at (also updates local_range and max_length)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    token_idx = args.token_idx
    # local_range: 100 tokens before token_idx
    local_range = (max(0, token_idx - 100), token_idx)
    # max_length: ensure we have enough context (token_idx + buffer for generation)
    max_length = token_idx + 100

    output_file = f"attention_sink_results_tok{token_idx}.tsv"

    # Load the long_docs dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(str(dataset_path))
    print(f"Loaded {len(dataset)} documents")

    # Optionally limit number of documents
    num_docs = args.num_docs if args.num_docs else len(dataset)
    num_docs = min(num_docs, len(dataset))

    all_results = []

    for model_name in args.models:
        try:
            model = OlmoCoreModel(model_name=model_name, max_length=max_length)
        except Exception as e:
            print(f"Error on model load: {model_name}")
            print(getattr(e, 'message', str(e)))
            continue

        # Results for this model, grouped by layer across all documents
        # layer -> list of (percent_sink, percent_local, doc_id, is_sink_layer) tuples
        layer_results: dict[int, List[Tuple[float, float, str, int]]] = defaultdict(list)

        # Create and attach the monitor once per model
        monitor = AttentionSinkMonitor(
            token_idx=token_idx,
            sink_range=(0, 100),
            local_range=local_range,
        )
        monitor.attach(model.generation_module.model)

        for doc_idx in range(num_docs):
            # Reset monitor for this document (keeps hooks attached)
            monitor.reset()

            doc = dataset[doc_idx]
            input_text = doc['text']
            source = doc.get('source', f'doc_{doc_idx}')
            doc_id = f"{source}_{doc_idx}"

            inputs = model.tokenizer([input_text], return_tensors='pt', padding=False)
            seq_len = inputs['input_ids'].shape[1]

            if seq_len <= token_idx:
                print(f"Skipping doc {doc_idx} ({source}): only {seq_len} tokens, need > {token_idx}")
                continue

            print(f"Processing: {model_name} with doc {doc_idx} ({source}, {seq_len} tokens)")
            model.generation_module.model_forward(**inputs)

            # Collect results for this document
            for result in monitor.get_results():
                layer_results[result['layer']].append(
                    (result['percent_sink'], result['percent_local'], doc_id, result['is_sink_layer'])
                )

        # Detach hooks after processing all documents for this model
        monitor.detach()

        # Store individual and averaged results for this model
        for layer, results in sorted(layer_results.items()):
            # is_sink_layer is the same for all results in this layer
            is_sink_layer = results[0][3]

            # Individual results for each document
            for percent_sink, percent_local, doc_id, _ in results:
                all_results.append({
                    'model': model_name,
                    'layer': layer,
                    'doc_id': doc_id,
                    'percent_sink': percent_sink,
                    'percent_local': percent_local,
                    'is_sink_layer': is_sink_layer,
                })

            # Average across all documents for this layer
            avg_sink = sum(r[0] for r in results) / len(results)
            avg_local = sum(r[1] for r in results) / len(results)
            all_results.append({
                'model': model_name,
                'layer': layer,
                'doc_id': 'AVERAGE',
                'percent_sink': avg_sink,
                'percent_local': avg_local,
                'is_sink_layer': is_sink_layer,
            })

    # Write all results to TSV file
    with open(output_file, 'w') as f:
        # Write header
        f.write("model\tlayer\tdoc_id\tpercent_sink\tpercent_local\tis_sink_layer\n")

        # Write all results
        for result in all_results:
            f.write(
                f"{result['model']}\t"
                f"{result['layer']}\t"
                f"{result['doc_id']}\t"
                f"{result['percent_sink']:.2f}\t"
                f"{result['percent_local']:.2f}\t"
                f"{result['is_sink_layer']}\n"
            )

    print(f"\nResults written to: {output_file}")


if __name__ == "__main__":
    main()
