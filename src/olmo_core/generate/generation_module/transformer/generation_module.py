import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast
import math
import os

import torch
import torch.distributed as dist
from torch.nn import functional as F
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed.checkpoint.metadata import Metadata
from tqdm import tqdm

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.utils import get_labels
from olmo_core.data.tokenizer import TokenizerConfig, ByteTokenizer, ByteTokenizerConfig
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
from olmo_core.generate.utils import selective_log_softmax
from olmo_core.io import is_url, join_path, normalize_path
from olmo_core.nn.blt.config import BLTConfig
import olmo_core.nn.blt.utils as blt_utils
from olmo_core.nn.transformer import Transformer, TransformerConfig, TransformerType
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import TransformerDataParallelConfig
from olmo_core.utils import get_default_device, move_to_device

from ..generation_module import GenerationModule

log = logging.getLogger(__name__)

BYTE_EXPANSION_FACTOR = int(os.environ.get("BYTE_EXPANSION_FACTOR", "6"))

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
        # device_name = torch.cuda.get_device_name(self.device) DEBUG commented
        # assert "H100" in device_name, (
        #     f"Expected H100 GPU, but got {device_name}. Flash attention w/ kv caching is not verified to work on non-Hopper GPUs."
        # )

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
        return_logprobs: bool = False,
        completions_only: bool = False,
        log_timing: bool = True,
        **generation_kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate text using greedy decoding.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len). This should be
                a *left-padding* mask, not an arbitrary attention mask.
                If not provided, the model will use a default mask that attends to all previous tokens.
            return_logits: If True, return logits along with generated tokens
            return_logprobs: If True, return log probabilities of generated tokens. If logits are
                only required for the purpose of computing logprobs, then use this option instead
                of return_logits - it is more memory efficient.
            completions_only: If True, return only the completions, not the entire sequence
            **generation_kwargs: Generation configuration overrides
        Returns:
            Tuple of (generated_ids, logits, logprobs) where:
                - generated_ids: Generated token IDs of shape (batch_size, output_length)
                - logits: Full logits if return_logits=True, else None. Shape: (batch_size, output_length, vocab_size)
                - logprobs: Log probabilities of tokens if return_logprobs=True, else None.
                  Shape: (batch_size, output_length - 1). Note: log probabilities are only computed
                  for positions 1 to N since the first token has no previous context.
        """
        # TODO: try padding the input_ids to a multiple of 128
        self.model.eval()
        input_ids = move_to_device(input_ids, self.device)
        if attention_mask is not None:
            attention_mask = move_to_device(attention_mask, self.device)
        batch_size, prompt_len = input_ids.shape
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Replace generation config with any overrides.
        generation_config = self._generation_config.replace(**generation_kwargs)

        # Output containers
        generated = input_ids
        all_logits = [] if return_logits else None
        all_logprobs = [] if return_logprobs else None

        # Timing stats
        start_time = time.perf_counter()
        time_to_first_token = None
        token_times = []
        tokens_generated = 0
        kv_cache_init_time = None
        prefill_time = None
        first_decode_time = None

        # Memory tracking
        initial_memory = None
        prefill_peak_memory = None
        decoding_peak_memory = None
        post_prefill_memory = None
        if self.device.type == "cuda" and log_timing:
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_max_memory_allocated(self.device)

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

        # Initialize/Reset the KV cache
        kv_cache_start_time = time.perf_counter()
        if generation_config.use_cache:
            self.reset_kv_cache(
                generation_config.use_cache, batch_size, max_length, self.model.dtype
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        kv_cache_init_time = time.perf_counter() - kv_cache_start_time

        pbar = tqdm(desc="Generating tokens", disable=not log_timing)
        while True:
            token_start_time = time.perf_counter()

            # Determine model inputs based on whether we're using the cache
            is_first_forward = generated.shape[1] == prompt_len
            is_using_cache = generation_config.use_cache and not is_first_forward

            input_ids_for_model = generated[:, -1:] if is_using_cache else generated
            attention_mask_for_model = None if is_using_cache else attention_mask
            logits_to_keep = (
                1 if (is_using_cache or completions_only) else prompt_len
            )  # Todo: we also dont need full logits unless we're returning them

            # Time the forward pass
            forward_start_time = time.perf_counter()
            logits = self.model(  # (batch_size, generated.shape[1], self.model.vocab_size)
                input_ids_for_model,
                attention_mask=attention_mask_for_model,
                logits_to_keep=logits_to_keep,
                prefill_kv_cache=is_first_forward,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            forward_end_time = time.perf_counter()

            # Track prefill and first decode times
            if is_first_forward:
                prefill_time = forward_end_time - forward_start_time
            elif tokens_generated == 0:
                first_decode_time = forward_end_time - forward_start_time

            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Track memory after prefill phase
            if is_first_forward and self.device.type == "cuda" and log_timing:
                torch.cuda.synchronize()
                prefill_peak_memory = torch.cuda.max_memory_allocated(self.device)
                post_prefill_memory = torch.cuda.memory_allocated(self.device)
                # Reset max memory tracking for decoding phase
                torch.cuda.reset_max_memory_allocated(self.device)

            if all_logits is not None:
                if is_first_forward and prompt_len > 1 and not completions_only:
                    all_logits.append(logits[:, :-1, :])
                all_logits.append(next_token_logits.unsqueeze(1))

            # Check if we should stop before generating more tokens
            pbar.update(1)
            if (max_length is not None and generated.shape[1] >= max_length) or finished.all():
                pbar.close()
                break

            # Select next tokens
            next_tokens = select_next_token(  # (batch_size,)
                next_token_logits,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            if all_logprobs is not None:
                if is_first_forward and prompt_len > 1 and not completions_only:
                    prompt_log_probs = selective_log_softmax(logits[:, :-1, :], generated[:, 1:])
                    all_logprobs.append(prompt_log_probs)
                next_token_log_prob = selective_log_softmax(next_token_logits, next_tokens)
                all_logprobs.append(next_token_log_prob.unsqueeze(1))

            # Handle finished sequences
            # - Check which sequences have generated EOS or stop tokens
            prev_finished = finished.clone()
            just_finished = next_tokens == generation_config.eos_token_id

            # Also check for stop tokens if provided
            if generation_config.stop_token_ids is not None:
                for stop_token_id in generation_config.stop_token_ids:
                    just_finished |= next_tokens == stop_token_id

            finished |= just_finished

            # - Append next tokens
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)

            # - Update attention mask if provided
            if attention_mask is not None:
                new_token_mask = torch.ones_like(prev_finished, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, new_token_mask.unsqueeze(-1)], dim=1)

            # Track timing
            token_end_time = time.perf_counter()
            if time_to_first_token is None:
                time_to_first_token = token_end_time - start_time
            else:
                token_times.append(token_end_time - token_start_time)
            tokens_generated += 1

        # Track peak memory for decoding phase
        if self.device.type == "cuda" and log_timing and prefill_peak_memory is not None:
            torch.cuda.synchronize()
            decoding_peak_memory = torch.cuda.max_memory_allocated(self.device)

        logits = None
        if return_logits and all_logits:
            logits = torch.cat(all_logits, dim=1)
            # TODO: align logits with generated tokens? OBO?
        logprobs = None
        if return_logprobs and all_logprobs:
            logprobs = torch.cat(all_logprobs, dim=1)
            # TODO: align logprobs with generated tokens? OBO?

        if completions_only:
            generated = generated[:, prompt_len:]
            if logits is not None:
                logits = logits[:, prompt_len:, :]
            if logprobs is not None:
                logprobs = logprobs[:, prompt_len:]

        total_time = time.perf_counter() - start_time
        if log_timing:
            print("Generation stats:")
            print(f"  Batch size: {batch_size}  Prompt length: {prompt_len}")
            print(
                f"  Tokens generated: {tokens_generated * batch_size} ({tokens_generated} per sequence, "
                f"{tokens_generated * batch_size / total_time:.1f} tokens/s)"
            )
            print(f"  Sequence length extended: {prompt_len} → {prompt_len + tokens_generated}")
            print(f"  Total generation time: {total_time:.3f}s")

            # Detailed timing breakdown
            print("\n  Timing breakdown:")
            if kv_cache_init_time is not None:
                print(f"    KV cache initialization: {kv_cache_init_time * 1000:.1f}ms")
            if prefill_time is not None:
                print(
                    f"    Prefill time: {prefill_time * 1000:.1f}ms ({prompt_len / prefill_time:.1f} tokens/s)"
                )
            if first_decode_time is not None:
                print(f"    First decode time: {first_decode_time * 1000:.1f}ms")
            if time_to_first_token is not None:
                print(f"    Time to first token: {time_to_first_token:.3f}s")
            if token_times:
                avg_inter_token_latency = sum(token_times) / len(token_times)
                print(f"    Average inter-token latency: {avg_inter_token_latency * 1000:.1f}ms")
                if len(token_times) > 1:
                    # Exclude first decode from average since it's often slower
                    avg_subsequent_latency = sum(token_times[1:]) / len(token_times[1:])
                    print(
                        f"    Average subsequent token latency: {avg_subsequent_latency * 1000:.1f}ms"
                    )

            # Log GPU memory usage
            if self.device.type == "cuda" and initial_memory is not None:
                print("\n  GPU memory usage:")
                print(f"    Initial memory: {initial_memory / 1024**3:.2f} GB")

                if prefill_peak_memory is not None:
                    prefill_memory_used = prefill_peak_memory - initial_memory
                    print("    Prefill phase:")
                    print(f"      Peak memory: {prefill_peak_memory / 1024**3:.2f} GB")
                    print(f"      Memory used: {prefill_memory_used / 1024**3:.2f} GB")

                    if decoding_peak_memory is not None and post_prefill_memory is not None:
                        # decoding_peak_memory is measured after reset, so it's relative to post-prefill state
                        decoding_memory_used = decoding_peak_memory - (
                            post_prefill_memory - prefill_peak_memory
                        )
                        print("    Decoding phase:")
                        print(
                            f"      Peak memory: {(post_prefill_memory + decoding_memory_used) / 1024**3:.2f} GB"
                        )
                        print(
                            f"      Additional memory used: {decoding_memory_used / 1024**3:.2f} GB"
                        )

        return generated, logits, logprobs

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

        # TODO(benjaminm): this does not seem like a good place..
        if transformer_config.name == TransformerType.blt:  # type: ignore
            return BLTTransformerGenerationModule.from_checkpoint(
                checkpoint_dir=checkpoint_dir,
                transformer_config=transformer_config,
                generation_config=generation_config,
                process_group=process_group,
                work_dir=work_dir,
                pre_download=pre_download,
                load_thread_count=load_thread_count,
                dtype=dtype,
                **kwargs,
            )

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

class BLTTransformerGenerationModule(TransformerGenerationModule):
    def __init__(
        self,
        model: Transformer,
        tokenizer: ByteTokenizer,
        generation_config: GenerationConfig,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        device: Optional[torch.device] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            model=model,
            generation_config=generation_config,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            device=device,
            state_dict_load_opts=state_dict_load_opts,
            state_dict_save_opts=state_dict_save_opts,
            load_key_mapping=load_key_mapping,
        )
        self.tokenizer = tokenizer

        # temporary, need to find a better solution
        self.blt_config = BLTConfig(
            teacher_force_boundaries=False,
            boundary_threshold="sample:0",
            skip_teacher=True,
        )

    def reset_kv_cache(
        self,
        use_cache: bool,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        blocks = [
            *list(self.model.blocks.values()),
            *list(self.model.local_encoder.blocks.values()),  # type: ignore
            *list(self.model.local_decoder.blocks.values()),  # type: ignore
        ]

        # TODO(benjaminm): different max seq len for local and global
        for block in blocks:
            if hasattr(block, "attention"):
                if hasattr(block.attention, "reset_kv_cache"):
                    block.attention.reset_kv_cache(  # type: ignore
                        # flash attention needs bf16
                        use_cache=use_cache, batch_size=batch_size, max_seq_len=max_seq_len, dtype=torch.bfloat16,
                    )

            if hasattr(block, "mamba"):
                if hasattr(block.mamba, "reset_kv_cache"):
                    block.mamba.reset_kv_cache(  # type: ignore
                        use_cache=use_cache, batch_size=batch_size, max_seq_len=max_seq_len, dtype=dtype
                    )

        if hasattr(self.model.local_decoder, "reset_kv_cache"):
            self.model.local_decoder.reset_kv_cache(  # type: ignore
                use_cache=use_cache, batch_size=batch_size, max_seq_len=max_seq_len, dtype=dtype
            )

    def free_kv_cache(self):
        blocks = [
            *list(self.model.blocks.values()),
            *list(self.model.local_encoder.blocks.values()),  # type: ignore
            *list(self.model.local_decoder.blocks.values()),  # type: ignore
        ]

        for block in blocks:
            if hasattr(block, "attention"):
                if hasattr(block.attention, "free_kv_cache"):
                    block.attention.free_kv_cache()

            if hasattr(block, "mamba"):
                if hasattr(block.mamba, "free_kv_cache"):
                    block.mamba.free_kv_cache()

        if hasattr(self.model.local_decoder, "free_kv_cache"):
            self.model.local_decoder.free_kv_cache()  # type: ignore

    @torch.inference_mode()
    def model_forward(self, input_ids: torch.Tensor, **kwargs):
        self.model.eval()
        input_ids = move_to_device(input_ids, self.device)
        
        # dummy patch lengths - not used
        patch_lens = torch.ones_like(input_ids)

        labels = F.pad(
            input_ids[..., 1:], (0, 1, 0, 0), value=-100
        )

        return self.model.student_forward(  # type: ignore
            input_ids,
            labels=labels,
            patch_lens=patch_lens,
            blt_config=self.blt_config,
            **kwargs
        )[0].logits

    @torch.inference_mode()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        return_logprobs: bool = True,
        completions_only: bool = False,
        log_timing: bool = True,
        stream: bool = False,
        **generation_kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate text using greedy decoding.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len). This should be
                a *left-padding* mask, not an arbitrary attention mask.
                If not provided, the model will use a default mask that attends to all previous tokens.
            return_logits: If True, return logits along with generated tokens
            return_logprobs: If True, return log probabilities of generated tokens. If logits are
                only required for the purpose of computing logprobs, then use this option instead
                of return_logits - it is more memory efficient.
            completions_only: If True, return only the completions, not the entire sequence
            **generation_kwargs: Generation configuration overrides
        Returns:
            Tuple of (generated_ids, logits, logprobs) where:
                - generated_ids: Generated token IDs of shape (batch_size, output_length)
                - logits: Full logits if return_logits=True, else None. Shape: (batch_size, output_length, vocab_size)
                - logprobs: Log probabilities of tokens if return_logprobs=True, else None.
                  Shape: (batch_size, output_length - 1). Note: log probabilities are only computed
                  for positions 1 to N since the first token has no previous context.
        """
        # convert to byte ids and compute patch lengths
        # support only bs=1 for now.
        assert input_ids.shape[0] == 1, "Only batch size of 1 is supported for BLT"
        attention_mask = None # not needed for bs=1

        byte_input_ids, patch_lens = self.tokenizer.get_tokens_and_patch_lengths(input_ids[0].tolist())

        byte_input_ids = torch.tensor([byte_input_ids], dtype=torch.long, device=input_ids.device)
        # TODO(benjaminm): our 'budget' of global tokens. we set it to the worst case (n_tokens == n_bytes)
        # but this is probably quite inefficient. do better?
        patch_lens = torch.ones_like(byte_input_ids, dtype=torch.long, device=input_ids.device)

        # TODO: try padding the input_ids to a multiple of 128
        self.model.eval()
        byte_input_ids = move_to_device(byte_input_ids, self.device)
        if attention_mask is not None:
            attention_mask = move_to_device(attention_mask, self.device)
        batch_size, prompt_len = byte_input_ids.shape
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Replace generation config with any overrides.
        generation_config = self._generation_config.replace(**generation_kwargs)

        # Output containers
        generated = byte_input_ids
        all_logits = [] if return_logits else None
        all_logprobs = [] if return_logprobs else None

        # Timing stats
        start_time = time.perf_counter()
        time_to_first_token = None
        token_times = []
        tokens_generated = 0
        kv_cache_init_time = None
        prefill_time = None
        first_decode_time = None

        # Memory tracking
        initial_memory = None
        prefill_peak_memory = None
        decoding_peak_memory = None
        post_prefill_memory = None
        if self.device.type == "cuda" and log_timing:
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_max_memory_allocated(self.device)

        # TODO(benjaminm): should probably get rid of BYTE_EXPANSION_FACTOR and handle length diff on the caller side
        if generation_config.max_new_tokens is not None:
            max_length = prompt_len + generation_config.max_new_tokens * BYTE_EXPANSION_FACTOR
        elif generation_config.max_length is not None:
            max_length = generation_config.max_length * BYTE_EXPANSION_FACTOR
        elif generation_config.use_cache:
            raise OLMoConfigurationError(
                "max_length or max_new_tokens must be provided if use_cache is True"
            )
        else:
            max_length = None  # Generate until EOS or stop tokens or OOM...

        # Initialize/Reset the KV cache
        kv_cache_start_time = time.perf_counter()
        if generation_config.use_cache:
            self.reset_kv_cache(
                generation_config.use_cache, batch_size, max_length, self.model.dtype
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        kv_cache_init_time = time.perf_counter() - kv_cache_start_time

        pbar = tqdm(desc="Generating tokens", disable=True)
        while True:
            token_start_time = time.perf_counter()

            # Determine model inputs based on whether we're using the cache
            is_first_forward = generated.shape[1] == prompt_len
            is_using_cache = generation_config.use_cache and not is_first_forward

            input_ids_for_model = generated[:, -1:] if is_using_cache else generated
            attention_mask_for_model = None if is_using_cache else attention_mask
            logits_to_keep = (
                1 if (is_using_cache or completions_only) else prompt_len
            )  # Todo: we also dont need full logits unless we're returning them

            patch_lens_for_model = torch.tensor(
                [[1]],
                device=patch_lens.device,
                dtype=patch_lens.dtype
            ) if is_using_cache else patch_lens

            # Time the forward pass
            forward_start_time = time.perf_counter()
            logits, last_token_is_boundary = self.model(  # (batch_size, generated.shape[1], self.model.vocab_size)
                input_ids_for_model,
                attention_mask=attention_mask_for_model,
                logits_to_keep=logits_to_keep,
                patch_lens=patch_lens_for_model,
                prefill_kv_cache=is_first_forward,
                blt_config=self.blt_config,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            forward_end_time = time.perf_counter()

            # Track prefill and first decode times
            if is_first_forward:
                prefill_time = forward_end_time - forward_start_time
            elif tokens_generated == 0:
                first_decode_time = forward_end_time - forward_start_time

            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Track memory after prefill phase
            if is_first_forward and self.device.type == "cuda" and log_timing:
                torch.cuda.synchronize()
                prefill_peak_memory = torch.cuda.max_memory_allocated(self.device)
                post_prefill_memory = torch.cuda.memory_allocated(self.device)
                # Reset max memory tracking for decoding phase
                torch.cuda.reset_max_memory_allocated(self.device)

            if all_logits is not None:
                if is_first_forward and prompt_len > 1 and not completions_only:
                    all_logits.append(logits[:, :-1, :])
                all_logits.append(next_token_logits.unsqueeze(1))

            # Check if we should stop before generating more tokens
            pbar.update(1)
            if (max_length is not None and generated.shape[1] >= max_length) or finished.all():
                pbar.close()
                break

            # Select next tokens
            next_tokens = select_next_token(  # (batch_size,)
                next_token_logits,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            if all_logprobs is not None:
                if is_first_forward and prompt_len > 1 and not completions_only:
                    prompt_log_probs = selective_log_softmax(logits[:, :-1, :], generated[:, 1:])
                    all_logprobs.append(prompt_log_probs)
                next_token_log_prob = selective_log_softmax(next_token_logits, next_tokens)
                all_logprobs.append(next_token_log_prob.unsqueeze(1))

            # Handle finished sequences
            # - Check which sequences have generated EOS or stop tokens
            prev_finished = finished.clone()
            just_finished = next_tokens == generation_config.eos_token_id

            # Also check for stop tokens if provided
            if generation_config.stop_token_ids is not None:
                for stop_token_id in generation_config.stop_token_ids:
                    just_finished |= next_tokens == stop_token_id

            finished |= just_finished

            # - Append next tokens
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)

            # - Update patch lengths if necessary
            if last_token_is_boundary:
                patch_lens = torch.cat([patch_lens, torch.tensor([[1]], device=self.device)], dim=1)
            else:
                patch_lens[:, -1] += 1

            # - Update attention mask if provided
            if attention_mask is not None:
                new_token_mask = torch.ones_like(prev_finished, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, new_token_mask.unsqueeze(-1)], dim=1)

            # Track timing
            token_end_time = time.perf_counter()
            if time_to_first_token is None:
                time_to_first_token = token_end_time - start_time
            else:
                token_times.append(token_end_time - token_start_time)
            tokens_generated += 1

            if stream:
                RED = "\033[0;31m"
                GREEN = "\033[0;32m"
                RESET = "\033[0;0m"

                tokens_to_print = generated[0][:-1].tolist() if tokens_generated == 1 else generated[0][-2:-1].tolist()

                if last_token_is_boundary and tokens_generated > 1:
                    print(RED + self.tokenizer.decode(tokens_to_print) + RESET, end="", flush=True)
                else:
                    if tokens_generated > 1:
                        print(GREEN + self.tokenizer.decode(tokens_to_print) + RESET, end="", flush=True)
                    else:
                        print(self.tokenizer.decode(tokens_to_print), end="", flush=True)

        if stream:
            print()

        # Track peak memory for decoding phase
        if self.device.type == "cuda" and log_timing and prefill_peak_memory is not None:
            torch.cuda.synchronize()
            decoding_peak_memory = torch.cuda.max_memory_allocated(self.device)

        logits = None
        if return_logits and all_logits:
            logits = torch.cat(all_logits, dim=1)
            # TODO: align logits with generated tokens? OBO?

            # check if greedy
            self.free_kv_cache()
            with torch.no_grad():
                reference_out, _ = self.model(
                    input_ids=generated,
                    labels=get_labels({"input_ids": generated}),
                    patch_lens=patch_lens,
                    blt_config=self.blt_config,
                )
            # last logit is potentially wrong since last_token_is_boundary=True in reference, so skip it
            assert torch.argmax(logits[:, -tokens_generated-1:-1], -1) == reference_out.logits[:, -tokens_generated-1:-1].argmax(-1)
            assert torch.allclose(logits[:, -tokens_generated-1:-1], reference_out.logits[:, -tokens_generated-1:-1], rtol=1e-1, atol=1)
        logprobs = None
        if return_logprobs and all_logprobs:
            logprobs = torch.cat(all_logprobs, dim=1)
            # TODO: align logprobs with generated tokens? OBO?

        if completions_only:
            generated = generated[:, prompt_len:]
            if logits is not None:
                logits = logits[:, prompt_len:, :]
            if logprobs is not None:
                logprobs = logprobs[:, prompt_len:]

        total_time = time.perf_counter() - start_time
        if log_timing:
            print("Generation stats:")
            print(f"  Batch size: {batch_size}  Prompt length: {prompt_len}")
            print(
                f"  Tokens generated: {tokens_generated * batch_size} ({tokens_generated} per sequence, "
                f"{tokens_generated * batch_size / total_time:.1f} tokens/s)"
            )
            print(f"  Sequence length extended: {prompt_len} → {prompt_len + tokens_generated}")
            print(f"  Total generation time: {total_time:.3f}s")

            # Detailed timing breakdown
            print("\n  Timing breakdown:")
            if kv_cache_init_time is not None:
                print(f"    KV cache initialization: {kv_cache_init_time * 1000:.1f}ms")
            if prefill_time is not None:
                print(
                    f"    Prefill time: {prefill_time * 1000:.1f}ms ({prompt_len / prefill_time:.1f} tokens/s)"
                )
            if first_decode_time is not None:
                print(f"    First decode time: {first_decode_time * 1000:.1f}ms")
            if time_to_first_token is not None:
                print(f"    Time to first token: {time_to_first_token:.3f}s")
            if token_times:
                avg_inter_token_latency = sum(token_times) / len(token_times)
                print(f"    Average inter-token latency: {avg_inter_token_latency * 1000:.1f}ms")
                if len(token_times) > 1:
                    # Exclude first decode from average since it's often slower
                    avg_subsequent_latency = sum(token_times[1:]) / len(token_times[1:])
                    print(
                        f"    Average subsequent token latency: {avg_subsequent_latency * 1000:.1f}ms"
                    )

            # Log GPU memory usage
            if self.device.type == "cuda" and initial_memory is not None:
                print("\n  GPU memory usage:")
                print(f"    Initial memory: {initial_memory / 1024**3:.2f} GB")

                if prefill_peak_memory is not None:
                    prefill_memory_used = prefill_peak_memory - initial_memory
                    print("    Prefill phase:")
                    print(f"      Peak memory: {prefill_peak_memory / 1024**3:.2f} GB")
                    print(f"      Memory used: {prefill_memory_used / 1024**3:.2f} GB")

                    if decoding_peak_memory is not None and post_prefill_memory is not None:
                        # decoding_peak_memory is measured after reset, so it's relative to post-prefill state
                        decoding_memory_used = decoding_peak_memory - (
                            post_prefill_memory - prefill_peak_memory
                        )
                        print("    Decoding phase:")
                        print(
                            f"      Peak memory: {(post_prefill_memory + decoding_memory_used) / 1024**3:.2f} GB"
                        )
                        print(
                            f"      Additional memory used: {decoding_memory_used / 1024**3:.2f} GB"
                        )

        # convert to token-level ids / logits / logprobs
        generated_continuation = generated[:, prompt_len:]

        # decoding errors could be a problem here - but not really a way around
        generated_text = self.tokenizer.decode(generated_continuation[0].tolist())
        generated_subword_ids = torch.tensor([self.tokenizer.hf_tokenizer.encode(generated_text)], dtype=torch.int64, device=self.device)
        _, patch_lens = self.tokenizer.get_tokens_and_patch_lengths(generated_subword_ids[0].tolist(), add_bos=False)
        patch_lens = torch.tensor([patch_lens], dtype=torch.int32, device=self.device)
        patch_ids = blt_utils.lengths_to_ids(patch_lens, generated_continuation.shape[-1]).to(self.device)

        if return_logits or return_logprobs:
            assert logits is not None and logprobs is not None

            main_path_patch_logprobs = torch.zeros((1, generated_subword_ids.shape[1]), device=self.device, dtype=torch.float32)
            main_path_patch_logprobs = main_path_patch_logprobs.scatter_reduce(
                src=logprobs.float(),
                dim=1,
                index=patch_ids,
                reduce="sum",
                include_self=False,
            )

            # we can't compute token-level logits, but logprobs should be fine since F.log_softmax(logprobs, -1) == logprobs
            patch_logits = torch.zeros((1, generated_subword_ids.shape[1] + 1, len(self.tokenizer.hf_tokenizer)), device=self.device, dtype=torch.float32)
            remaining_logpmass = blt_utils.log1mexp(main_path_patch_logprobs)
            remaining_logp_uniform = remaining_logpmass - math.log(patch_logits.shape[2] - 1)  # -1 to skip the main path token

            patch_logits[:, :-1, :] = remaining_logp_uniform.unsqueeze(-1)
            patch_logits.scatter_(
                -1,
                generated_subword_ids.unsqueeze(-1),
                main_path_patch_logprobs.to(patch_logits.dtype).unsqueeze(-1),
            )
        else:
            main_path_patch_logprobs = None
            patch_logits = None

        return generated_subword_ids, patch_logits, main_path_patch_logprobs

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
        if transformer_config is None or tokenizer_config is None and get_rank(process_group) == 0:
            config_path = join_path(checkpoint_dir, "config.json")
            with cached_path(config_path).open() as f:
                config_dict = json.load(f)
            try:
                # Avoid loading the entire experiment config b/c we don't care about validation outside
                # of the transformer config and the tokenizer config
                if transformer_config is None:
                    transformer_config = TransformerConfig.from_dict(config_dict["model"])
                if tokenizer_config is None:
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

        # DEBUG force enable flash attention
        transformer_config.block.attention.use_flash = True

        print(transformer_config)
        print(generation_config)
        model = transformer_config.build()

        # DEBUG flash attention needs bf16
        for block in model.blocks.values():
            block.to(torch.bfloat16)

        generation_module = cls(model, cast(ByteTokenizerConfig, tokenizer_config).build(), generation_config, **kwargs)

        # Load checkpoint
        generation_module.load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            process_group=process_group,
            work_dir=work_dir,
            pre_download=pre_download,
            load_thread_count=load_thread_count,
        )

        return generation_module