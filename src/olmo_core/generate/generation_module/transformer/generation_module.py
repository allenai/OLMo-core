import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from rich import print
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed.checkpoint.metadata import Metadata
from tqdm import tqdm

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data.utils import attention_mask_to_cache_leftpad
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_state_dict
from olmo_core.distributed.parallel import build_world_mesh, get_dp_process_group
from olmo_core.distributed.utils import (
    get_fs_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    scatter_object,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.generate.sampling import select_next_token
from olmo_core.io import is_url, join_path, normalize_path
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import TransformerDataParallelConfig
from olmo_core.utils import get_default_device, move_to_device

from ..generation_module import GenerationModule

log = logging.getLogger(__name__)


class TransformerGenerationModule(GenerationModule):
    """Module for autoregressive text generation with transformer models."""

    def __init__(
        self,
        model: Transformer,
        generation_config: GenerationConfig,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        device: Optional[torch.device] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()

        # Build world mesh.
        self.device = device or get_default_device()

        # Verify H100 GPU
        assert self.device.type == "cuda", f"Expected CUDA device, got {self.device.type}"
        device_name = torch.cuda.get_device_name(self.device)
        assert "H100" in device_name, (
            f"Expected H100 GPU, but got {device_name}. Flash attention w/ kv caching is not verified to work on non-Hopper GPUs."
        )

        self.world_mesh: Optional[DeviceMesh] = None
        if is_distributed():
            self.world_mesh = build_world_mesh(dp=dp_config, device_type=self.device.type)
        elif dp_config is not None:
            raise OLMoConfigurationError(
                "Training parallelism configs are only valid for distributed training"
            )

        # Parallelize model.
        self.model = parallelize_model(
            model,
            world_mesh=self.world_mesh,
            device=self.device,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
        )

        self._dp_config = dp_config
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(strict=True)
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(strict=True)
        self.load_key_mapping = load_key_mapping
        self._generation_config = generation_config

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return None if self.world_mesh is None else get_dp_process_group(self.world_mesh)

    @property
    def world_size(self) -> int:
        return get_world_size()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=self.state_dict_save_opts),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        dist_cp_sd.set_model_state_dict(
            self.model, state_dict["model"], options=self.state_dict_load_opts
        )

    def reset_kv_cache(
        self,
        use_cache: bool,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        for block in self.model.blocks.values():
            if hasattr(block.attention, "reset_kv_cache"):
                block.attention.reset_kv_cache(  # type: ignore
                    use_cache=use_cache, batch_size=batch_size, max_seq_len=max_seq_len, dtype=dtype
                )

    def free_kv_cache(self):
        for block in self.model.blocks.values():
            if hasattr(block.attention, "free_kv_cache"):
                block.attention.free_kv_cache()  # type: ignore

    @torch.inference_mode()
    def model_forward(self, input_ids: torch.Tensor, **kwargs):
        self.model.eval()
        input_ids = move_to_device(input_ids, self.device)
        return self.model(input_ids, **kwargs)

    @torch.inference_mode()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        completions_only: bool = False,
        log_timing: bool = True,
        _forced_next_tokens: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate text using greedy decoding.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len). This should be
                a *left-padding* mask, not an arbitrary attention mask. If not provided, the model
                will assume there are no left-padding tokens.
            return_logits: If True, return logits along with generated tokens
            completions_only: If True, return only the completions, not the entire sequence
            **generation_kwargs: Generation configuration overrides
        Returns:
            Tuple of (generated_ids, logits, logprobs) where:
                - generated_ids: Generated token IDs of shape (batch_size, output_length)
                - logits: Full logits if return_logits=True, else None. Shape: (batch_size, output_length, vocab_size)
        """
        # Replace generation config with any overrides.
        generation_config = self._generation_config.replace(**generation_kwargs)
        eos = generation_config.eos_token_id

        self.model.eval()

        input_ids = move_to_device(input_ids, self.device)
        batch_size, prompt_len = input_ids.shape
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        stop_tokens = (
            torch.tensor(generation_config.stop_token_ids, device=self.device, dtype=torch.int32)
            if generation_config.stop_token_ids is not None
            else None
        )

        # Optional testing hook to force the generated token at each step.
        # Accepts shape (steps,) or (batch_size, steps). If provided, we will
        # override the next token with the forced token for the corresponding step.
        forced_tokens: Optional[torch.Tensor] = None
        if _forced_next_tokens is not None:
            if _forced_next_tokens.dim() not in (1, 2):
                raise ValueError(
                    "_forced_next_tokens must be 1D (steps,) or 2D (batch_size, steps)"
                )
            if _forced_next_tokens.dim() == 2 and _forced_next_tokens.shape[0] != batch_size:
                raise ValueError(
                    "_forced_next_tokens with 2D shape must have first dim equal to batch_size"
                )
            forced_tokens = move_to_device(_forced_next_tokens.to(torch.long), self.device)

        # Output containers
        generated = input_ids
        all_logits = [] if return_logits else None

        # Timing stats
        start_time = time.perf_counter()
        time_to_first_token = None
        token_times = []
        tokens_generated = 0
        kv_cache_init_time = None
        prefill_time = None

        if generation_config.max_new_tokens is not None:
            max_length = prompt_len + generation_config.max_new_tokens
        elif generation_config.max_length is not None:
            max_length = generation_config.max_length
        elif generation_config.use_cache:
            raise OLMoConfigurationError(
                "max_length or max_new_tokens must be provided if use_cache is True"
            )
        else:
            max_length = None  # Generate until EOS or stop tokens or OOM...

        prefill_seq_lens = None
        decode_seq_lens = None
        prefill_cache_leftpad = None
        if generation_config.use_cache:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)
            prefill_cache_leftpad, prefill_seq_lens = attention_mask_to_cache_leftpad(
                attention_mask
            )
            prefill_cache_leftpad = prefill_cache_leftpad.to(self.device)
            prefill_seq_lens = prefill_seq_lens.to(self.device).to(torch.int32)
            decode_seq_lens = torch.ones(batch_size, dtype=torch.int32, device=self.device)

        # Initialize/Reset the KV cache
        kv_cache_start_time = time.perf_counter()
        self.reset_kv_cache(generation_config.use_cache, batch_size, max_length, self.model.dtype)
        if self.device.type == "cuda" and log_timing:
            torch.cuda.synchronize()
        kv_cache_init_time = time.perf_counter() - kv_cache_start_time

        pbar = tqdm(
            desc="Generating tokens",
            unit="tokens",
            total=(max_length - prompt_len) if max_length is not None else None,
            disable=not log_timing,
            miniters=10,
            color="blue",
        )
        while not ((max_length is not None and generated.shape[1] >= max_length) or finished.all()):
            token_start_time = time.perf_counter()

            # Determine model inputs based on if we are prefilling or decoding
            is_first_forward = generated.shape[1] == prompt_len
            input_ids_for_model = (
                generated
                if (is_first_forward or not generation_config.use_cache)
                else generated[:, -1:]
            )
            step_seq_lens = prefill_seq_lens if is_first_forward else decode_seq_lens
            cache_leftpad = (
                prefill_cache_leftpad if is_first_forward and generation_config.use_cache else None
            )

            # Time the forward pass
            forward_start_time = time.perf_counter()
            next_token_logits = self.model(
                input_ids_for_model,
                logits_to_keep=1,
                use_cache=generation_config.use_cache,
                cache_leftpad=cache_leftpad if generation_config.use_cache else None,
                seq_lens=step_seq_lens if generation_config.use_cache else None,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            forward_end_time = time.perf_counter()

            # Track prefill and first decode times
            if is_first_forward:
                prefill_time = forward_end_time - forward_start_time

            if all_logits is not None:
                all_logits.append(next_token_logits)

            # Generate next token (predicted by sampling/argmax)
            predicted_next_tokens = select_next_token(  # (batch_size,)
                next_token_logits.squeeze(1),
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            # Optionally override with forced tokens for testing determinism across runs
            if forced_tokens is not None and (
                (forced_tokens.dim() == 1 and tokens_generated < forced_tokens.shape[0])
                or (forced_tokens.dim() == 2 and tokens_generated < forced_tokens.shape[1])
            ):
                if forced_tokens.dim() == 1:
                    next_tokens = forced_tokens[tokens_generated].expand(batch_size)
                else:
                    next_tokens = forced_tokens[:, tokens_generated]
            else:
                next_tokens = predicted_next_tokens

            # Force EOS for (previously) finished sequences
            next_tokens = torch.where(finished, torch.full_like(next_tokens, eos), next_tokens)

            # Handle finished sequences
            stop_hit = next_tokens.eq(eos)
            if stop_tokens is not None:
                stop_hit |= next_tokens.unsqueeze(-1).eq(stop_tokens).any(dim=-1)
            finished |= stop_hit

            # Append next tokens
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)

            # Track timing
            token_end_time = time.perf_counter()
            if time_to_first_token is None:
                time_to_first_token = token_end_time - start_time
            else:
                token_times.append(token_end_time - token_start_time)

            tokens_generated += 1
            pbar.update(1)

        pbar.close()

        logits = None
        if return_logits and all_logits:
            logits = torch.cat(all_logits, dim=1)
            expected_logits = generated.shape[1] - prompt_len
            assert logits.shape[1] == expected_logits, (
                f"Number of logits ({logits.shape[1]}) does not match number of newly generated tokens ({expected_logits})"
            )

        if completions_only:
            generated = generated[:, prompt_len:]
            # NOTE: completions_only does not apply to logits. They are already only computed for completions.

        total_time = time.perf_counter() - start_time
        if log_timing:
            # Calculate metrics
            total_tokens = tokens_generated * batch_size
            tokens_per_sec_total = total_tokens / total_time
            tokens_per_sec_per_seq = tokens_generated / total_time

            # Main generation stats
            print(f"\n{'=' * 60}\nGENERATION STATISTICS\n{'=' * 60}")
            print(f"  Batch size: {batch_size:,} | Prompt length: {prompt_len:,} tokens")
            print(
                f"  Tokens generated: {tokens_generated:,} per sequence | Total: {total_tokens:,}"
            )
            print(f"  Sequence length: {prompt_len:,} → {prompt_len + tokens_generated:,}")
            print(f"  Total generation time: {total_time:.3f}s")
            # Calculate prefill and completion throughput
            prefill_tokens_total = prompt_len * batch_size
            completion_tokens_total = tokens_generated * batch_size

            print("  Throughput:")
            print(
                f"    Overall: {tokens_per_sec_total:.1f} tokens/s (total) | {tokens_per_sec_per_seq:.1f} tokens/s (per seq)"
            )

            if prefill_time is not None and prefill_time > 0:
                prefill_throughput_total = prefill_tokens_total / prefill_time
                prefill_throughput_per_seq = prompt_len / prefill_time
                print(
                    f"    Prefill: {prefill_throughput_total:.1f} tokens/s (total) | {prefill_throughput_per_seq:.1f} tokens/s (per seq)"
                )

            completion_time = total_time - (prefill_time or 0) - (kv_cache_init_time or 0)
            if completion_time > 0 and tokens_generated > 0:
                completion_throughput_total = completion_tokens_total / completion_time
                completion_throughput_per_seq = tokens_generated / completion_time
                print(
                    f"    Completion: {completion_throughput_total:.1f} tokens/s (total) | {completion_throughput_per_seq:.1f} tokens/s (per seq)"
                )
            print(f"{'=' * 60}\n")

        return generated, logits

    def load_checkpoint(
        self,
        checkpoint_dir: PathOrStr,
        work_dir: PathOrStr,
        process_group: Optional[ProcessGroup] = None,
        pre_download: bool = True,
        load_thread_count: Optional[int] = None,
    ):
        """
        Load model checkpoint.

        Args:
            checkpoint_dir: Path to checkpoint directory
            work_dir: Working directory for caching remote checkpoints
            process_group: Process group for distributed loading
            pre_download: Whether to pre-download remote checkpoints
            load_thread_count: Number of threads to use for loading the checkpoint

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        work_dir = Path(work_dir)
        if get_fs_local_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = normalize_path(checkpoint_dir)
        train_module_dir = join_path(checkpoint_dir, "model_and_optim")
        metadata: Optional[Metadata] = None
        if get_rank(process_group) == 0:
            try:
                metadata = get_checkpoint_metadata(train_module_dir)
            except FileNotFoundError:
                # Try base directory, which could be the case if user is trying to
                # load model weights and not an actual train checkpoint.
                log.warning(
                    f"Checkpoint metadata not found in '{train_module_dir}', "
                    f"trying base directory '{checkpoint_dir}'"
                )
                metadata = get_checkpoint_metadata(checkpoint_dir)
                train_module_dir = checkpoint_dir

        train_module_dir = scatter_object(train_module_dir)
        if metadata is None:
            metadata = get_checkpoint_metadata(train_module_dir)

        state_dict = self.state_dict_to_load(metadata)
        load_state_dict(
            train_module_dir,
            state_dict,  # state is loaded into here, in-place
            process_group=process_group,
            pre_download=is_url(checkpoint_dir) and pre_download,
            work_dir=work_dir,
            thread_count=load_thread_count,
        )
        self.load_state_dict(state_dict)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: PathOrStr,
        *,
        transformer_config: Optional[TransformerConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
        process_group: Optional[ProcessGroup] = None,
        work_dir: Optional[PathOrStr] = None,
        pre_download: bool = True,
        load_thread_count: Optional[int] = None,
        dtype: Optional[DType] = None,
        **kwargs,
    ) -> "TransformerGenerationModule":
        """
        Create a GenerationModule from a checkpoint.

        This is a convenience method that combines model initialization and checkpoint loading.

        Args:
            checkpoint_dir: Path to checkpoint directory.
            transformer_config: Configuration for the transformer model. If not provided,
                will be loaded from the checkpoint's config.json file.
            generation_config: Configuration for generation. If not provided, uses default
                GenerationConfig.
            process_group: Process group for distributed checkpoint loading.
            pre_download: Whether to pre-download remote checkpoints.
            load_thread_count: Number of threads to use for loading checkpoint.
            dtype: If provided, build the model with this dtype.
            **kwargs: Additional keyword arguments passed to the TransformerGenerationModule
                constructor.

        Returns:
            TransformerGenerationModule instance with loaded checkpoint.

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist.
            OLMoConfigurationError: If transformer config cannot be determined.
            RuntimeError: If checkpoint loading fails.
        """
        checkpoint_dir = normalize_path(checkpoint_dir)

        # Load transformer config from checkpoint if not provided
        tokenizer_config = None
        if transformer_config is None and get_rank(process_group) == 0:
            config_path = join_path(checkpoint_dir, "config.json")
            with cached_path(config_path).open() as f:
                config_dict = json.load(f)
            try:
                # Avoid loading the entire experiment config b/c we don't care about validation outside
                # of the transformer config and the tokenizer config
                transformer_config = TransformerConfig.from_dict(config_dict["model"])
                tokenizer_config = TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])
            except KeyError as e:
                raise OLMoConfigurationError(
                    f"Failed to load config from checkpoint at {config_path}: missing required field {e}"
                ) from e

        # Create work directory on rank 0
        work_dir = Path(
            work_dir or (tempfile.mkdtemp() if get_rank(process_group) == 0 else "/tmp")
        )

        # Broadcast config and work_dir to all ranks
        transformer_config, work_dir, tokenizer_config = scatter_object(
            (transformer_config, work_dir, tokenizer_config)
        )

        if transformer_config is None:
            raise OLMoConfigurationError(
                "Transformer config is required. Either provide a transformer config or a "
                "checkpoint with a transformer config."
            )

        if generation_config is None:
            if tokenizer_config is None:
                raise OLMoConfigurationError(
                    "Tokenizer config is required to build generation config. Either provide a "
                    "generation config or a checkpoint with a tokenizer config."
                )
            generation_config = GenerationConfig(
                pad_token_id=tokenizer_config.pad_token_id,
                eos_token_id=tokenizer_config.eos_token_id,
            )
            log.info(
                f"No generation config provided, using defaults from checkpoint config: {generation_config}",
            )

        # Build model and generation module
        if dtype is not None:
            dtype = DType(dtype)
            transformer_config.apply(
                lambda c: setattr(c, "dtype", dtype) if hasattr(c, "dtype") else None
            )
        print(transformer_config)
        print(generation_config)
        model = transformer_config.build()
        generation_module = cls(model, generation_config, **kwargs)

        # Load checkpoint
        generation_module.load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            process_group=process_group,
            work_dir=work_dir,
            pre_download=pre_download,
            load_thread_count=load_thread_count,
        )

        return generation_module
