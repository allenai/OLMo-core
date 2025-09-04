import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import torch
from torch.nn import functional as F
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed.checkpoint.metadata import Metadata
from tqdm import tqdm

from olmo_core.data import ByteTokenizerConfig
from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.tokenizer import ByteTokenizer, TokenizerConfig
from olmo_core.data.utils import attention_mask_to_cache_leftpad, get_labels
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
from olmo_core.generate.generation_module import GenerationConfig, GenerationModule
from olmo_core.generate.sampling import select_next_token
from olmo_core.generate.utils import selective_log_softmax
from olmo_core.io import is_url, join_path, normalize_path
from olmo_core.nn.attention import Attention
from olmo_core.nn.blt.config import BLTConfig
import olmo_core.nn.blt.utils as blt_utils
from olmo_core.nn.mamba import Mamba
from olmo_core.nn.transformer import Transformer, TransformerConfig, TransformerType
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, log_or_print, move_to_device

log = logging.getLogger(__name__)

BYTE_EXPANSION_FACTOR = int(os.environ.get("BYTE_EXPANSION_FACTOR", "4"))

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

        self.device = device or get_default_device()
        if self.device.type != "cuda":
            raise AssertionError(f"Expected CUDA device, got {self.device.type}")

        device_name = torch.cuda.get_device_name(self.device)
        if "H100" not in device_name:
            log_or_print(
                log,
                "Flash attention w/ kv caching is not verified to work on non-Hopper GPUs.",
                level=logging.WARNING,
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
        self._model_mode: Optional[Literal["train", "eval"]] = None

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

    def prepare_inference_cache(self, batch_size: int, max_seq_len: int):
        # Note: not all models use key-value caches, which is why this method
        # is called "prepare_inference_cache" rather than "prepare_kv_cache".
        # For example, Mamba requires cache state but doesn't use a kv-cache.
        for block in self.model.blocks.values():
            assert isinstance(block.attention, Attention)
            attn = cast(Attention, block.attention)
            if attn.kv_cache_manager is None:
                attn.init_kv_cache_manager(batch_size, max_seq_len)
            else:
                attn.kv_cache_manager.reset(batch_size, max_seq_len)

    def free_inference_cache(self):
        for block in self.model.blocks.values():
            assert isinstance(block.attention, Attention)
            cast(Attention, block.attention).kv_cache_manager = None

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        if self._model_mode != mode:
            if mode == "train":
                self.model.train()
            elif mode == "eval":
                self.model.eval()
            else:
                raise ValueError(f"Invalid model mode: {mode}")
            self._model_mode = mode

    @torch.inference_mode()
    def model_forward(self, input_ids: torch.Tensor, **kwargs):
        self._set_model_mode("eval")
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
        Generate text with autoregressive decoding.

        :param input_ids: Input token IDs of shape ``(batch_size, seq_len)``.
        :param attention_mask: Optional attention mask of shape ``(batch_size, seq_len)``. This should be
            a *left-padding* mask, not an arbitrary attention mask. If not provided, the model
            will assume there are no left-padding tokens.
        :param return_logits: If ``True``, return logits along with generated tokens.
        :param return_logprobs: If ``True``, return log probabilities *for the generated tokens* along with
            generated tokens. This is notably more memory efficient than ``return_logits``.
        :param completions_only: If ``True``, return only the completions, not the entire sequence.
        :param generation_kwargs: Generation configuration overrides.

        :returns: Tuple of ``(generated_ids, logits, logprobs)`` where:
            - ``generated_ids``: Generated token IDs of shape ``(batch_size, output_length)``.
            - ``logits``: Full logits if ``return_logits=True``, else ``None``. Shape: ``(batch_size, output_length, vocab_size)``.
            - ``logprobs``: Log probabilities of generated tokens if ``return_logprobs=True``, else ``None``. Shape: ``(batch_size, output_length)``.
        """
        start_time = time.perf_counter()

        # Replace generation config with any overrides.
        generation_config = self._generation_config.replace(**generation_kwargs)
        eos = generation_config.eos_token_id
        pad = generation_config.pad_token_id

        input_ids = move_to_device(input_ids, self.device)
        batch_size, prompt_len = input_ids.shape
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        stop_tokens = (
            torch.tensor(generation_config.stop_token_ids, device=self.device, dtype=torch.int32)
            if generation_config.stop_token_ids is not None
            else None
        )

        # Output containers
        generated = input_ids
        all_logits: Optional[List[torch.Tensor]] = [] if return_logits else None
        all_logprobs: Optional[List[torch.Tensor]] = [] if return_logprobs else None

        # Timing stats
        time_to_first_token = None
        decode_start_time = None
        setup_time = None
        tokens_generated = 0

        if generation_config.max_new_tokens is not None:
            max_length = prompt_len + generation_config.max_new_tokens
        elif generation_config.max_length is not None:
            max_length = generation_config.max_length
        else:
            max_length = None  # Generate until EOS or stop tokens or OOM...

        prefill_cache_leftpad = None
        if generation_config.use_cache:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)
            prefill_cache_leftpad = attention_mask_to_cache_leftpad(attention_mask).to(self.device)

        # Initialize/Reset the inference cache
        if generation_config.use_cache:
            if max_length is None:
                raise OLMoConfigurationError(
                    "max_length or max_new_tokens must be provided if use_cache is True"
                )
            self.prepare_inference_cache(batch_size, max_length)
        else:
            self.free_inference_cache()

        pbar = tqdm(
            desc="Generating tokens",
            unit="tokens",
            total=(max_length - prompt_len) if max_length is not None else None,
            disable=not log_timing,
            miniters=10,
            colour="blue",
        )
        while not ((max_length is not None and generated.shape[1] >= max_length) or finished.all()):
            # Determine model inputs based on if we are prefilling or decoding
            is_first_forward = generated.shape[1] == prompt_len
            input_ids_for_model = (
                generated
                if (is_first_forward or not generation_config.use_cache)
                else generated[:, -1:]
            )
            cache_leftpad = (
                prefill_cache_leftpad if is_first_forward and generation_config.use_cache else None
            )

            # Forward pass - handles both prefill and decode phases
            forward_start_time = time.perf_counter()
            next_token_logits = self.model(  # (batch_size, seq_len=1, vocab_size)
                input_ids_for_model,
                logits_to_keep=1,
                cache_leftpad=cache_leftpad if generation_config.use_cache else None,
            )

            next_tokens = select_next_token(
                next_token_logits.squeeze(1),
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            if all_logits is not None:
                all_logits.append(next_token_logits)
            if all_logprobs is not None:
                all_logprobs.append(
                    selective_log_softmax(next_token_logits, next_tokens.unsqueeze(-1))
                )

            # Force EOS for (previously) finished sequences
            next_tokens = torch.where(finished, torch.full_like(next_tokens, eos), next_tokens)

            # Handle finished sequences
            stop_hit = next_tokens.eq(eos)
            if stop_tokens is not None:
                stop_hit |= next_tokens.unsqueeze(-1).eq(stop_tokens).any(dim=-1)
            finished |= stop_hit

            # Append next tokens
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)

            if log_timing and tokens_generated == 0:
                torch.cuda.synchronize()
                decode_start_time = time.perf_counter()
                time_to_first_token = decode_start_time - forward_start_time
                setup_time = forward_start_time - start_time
            tokens_generated += 1
            pbar.update(1)
        pbar.close()

        logits = logprobs = None
        if return_logits and all_logits:
            logits = torch.cat(all_logits, dim=1)
        if return_logprobs and all_logprobs:
            logprobs = torch.cat(all_logprobs, dim=1)

        if log_timing:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time = end_time - start_time
            total_tokens = generated.numel()
            prefill_tokens = prompt_len * batch_size
            completion_tokens = tokens_generated * batch_size
            tokens_per_sec_total = total_tokens / total_time
            tokens_per_sec_per_seq = tokens_generated / total_time
            pad_count = (generated == pad).sum().item()
            pad_percentage = (pad_count / total_tokens) * 100 if total_tokens > 0 else 0.0

            stats_lines = [
                f"\n{'=' * 60}",
                "GENERATION STATISTICS",
                f"  Batch size: {batch_size:,} | Prompt len: {prompt_len:,} tokens",
                f"  Tokens generated: {tokens_generated:,} per sequence | Total: {total_tokens:,}",
                f"  Seq length: {prompt_len:,} → {prompt_len + tokens_generated:,}",
                f"  Padding stats: {pad_count:,} / {total_tokens:,} ({pad_percentage:.1f}%)",
            ]
            if decode_start_time and forward_start_time and time_to_first_token:
                decode_time = end_time - decode_start_time
                completion_time = end_time - decode_start_time
                stats_lines.append(
                    f"  Throughput:\n"
                    f"    Setup: {setup_time:.3f}s | Prefill: {time_to_first_token:.3f}s | Decode: {decode_time:.3f}s | Total: {total_time:.3f}s\n"
                    f"    Overall TPS: {tokens_per_sec_per_seq:.1f} /seq | {tokens_per_sec_total:.1f} /total\n"
                    f"    Prefill TPS: {prompt_len / time_to_first_token:.1f} /seq | {prefill_tokens / time_to_first_token:.1f} /total\n"
                    f"    Completion TPS: {tokens_generated / completion_time:.1f} /seq | {completion_tokens / completion_time:.1f} /total",
                )

            stats_lines.append(f"{'=' * 60}")
            log_or_print(log, "\n".join(stats_lines))

        if completions_only:
            generated = generated[:, prompt_len:]
            # NOTE: completions_only does not apply to logits/logprobs. They are already computed only for completions.
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
                log_or_print(
                    log,
                    f"Checkpoint metadata not found in '{train_module_dir}', "
                    f"trying base directory '{checkpoint_dir}'",
                    level=logging.WARNING,
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
        if transformer_config.name in {TransformerType.blt, TransformerType.blt_distill}:  # type: ignore
            if transformer_config.name == TransformerType.blt_distill:  # type: ignore
                pass
                # don't need teacher
                transformer_config.teacher_config = None  # type: ignore

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
            log_or_print(
                log,
                f"No generation config provided, using defaults from checkpoint config: {generation_config}",
            )

        # Build model and generation module
        if dtype is not None:
            dtype = DType(dtype)
            transformer_config.apply(
                lambda c: setattr(c, "dtype", dtype) if hasattr(c, "dtype") else None
            )

        log_or_print(log, f"{transformer_config}")
        log_or_print(log, f"{generation_config}")
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
        blt_config: BLTConfig,
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
        self.blt_config = blt_config

    def prepare_inference_cache(self, batch_size: int, max_seq_len: int):
        blocks = [
            *list(self.model.blocks.values()),
            *list(self.model.local_encoder.blocks.values()),  # type: ignore
            *list(self.model.local_decoder.blocks.values()),  # type: ignore
        ]

        for block in blocks:
            if hasattr(block, "attention") and isinstance(block.attention, Attention):
                attn = block.attention
                if attn.kv_cache_manager is None:
                    attn.init_kv_cache_manager(batch_size, max_seq_len)
                else:
                    attn.kv_cache_manager.reset(batch_size, max_seq_len)
            elif hasattr(block, "mamba") and isinstance(block.mamba, Mamba):
                mamba = block.mamba
                if mamba.mamba_cache_manager is None:
                    mamba.init_mamba_cache_manager(batch_size)
                else:
                    mamba.mamba_cache_manager.reset(batch_size)

        self.model.local_encoder.prepare_inference_cache(batch_size, max_seq_len)  # type: ignore
        self.model.local_decoder.prepare_inference_cache(batch_size, max_seq_len)  # type: ignore

    def free_inference_cache(self):
        blocks = [
            *list(self.model.blocks.values()),
            *list(self.model.local_encoder.blocks.values()),  # type: ignore
            *list(self.model.local_decoder.blocks.values()),  # type: ignore
        ]

        for block in blocks:
            if hasattr(block, "attention") and isinstance(block.attention, Attention):
                block.attention.kv_cache_manager = None
            elif hasattr(block, "mamba") and isinstance(block.mamba, Mamba):
                block.mamba.mamba_cache_manager = None

        self.model.local_encoder.free_inference_cache()  # type: ignore
        self.model.local_decoder.free_inference_cache()  # type: ignore

    @torch.inference_mode()
    def model_forward(self, input_ids: torch.Tensor, **kwargs):
        raise NotImplementedError() # need to restore

        self._set_model_mode("eval")
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
        Generate text with autoregressive decoding.

        :param input_ids: Input token IDs of shape ``(batch_size, seq_len)``.
        :param attention_mask: Optional attention mask of shape ``(batch_size, seq_len)``. This should be
            a *left-padding* mask, not an arbitrary attention mask. If not provided, the model
            will assume there are no left-padding tokens.
        :param return_logits: If ``True``, return logits along with generated tokens.
        :param return_logprobs: If ``True``, return log probabilities *for the generated tokens* along with
            generated tokens. This is notably more memory efficient than ``return_logits``.
        :param completions_only: If ``True``, return only the completions, not the entire sequence.
        :param generation_kwargs: Generation configuration overrides.

        :returns: Tuple of ``(generated_ids, logits, logprobs)`` where:
            - ``generated_ids``: Generated token IDs of shape ``(batch_size, output_length)``.
            - ``logits``: Full logits if ``return_logits=True``, else ``None``. Shape: ``(batch_size, output_length, vocab_size)``.
            - ``logprobs``: Log probabilities of generated tokens if ``return_logprobs=True``, else ``None``. Shape: ``(batch_size, output_length)``.
        """
        start_time = time.perf_counter()

        # convert to byte ids and compute patch lengths
        # support only bs=1 for now.
        assert input_ids.shape[0] == 1, "Only batch size of 1 is supported for BLT"
        attention_mask = None # not needed for bs=1
        byte_input_ids, patch_lens = self.tokenizer.get_tokens_and_patch_lengths(input_ids[0].tolist(), add_bos=True)

        byte_input_ids = torch.tensor([byte_input_ids], dtype=torch.long, device=input_ids.device)
        patch_lens = torch.tensor([patch_lens], dtype=torch.long, device=input_ids.device)
        # TODO(benjaminm): our 'budget' of global tokens. we set it to the worst case (n_tokens == n_bytes)
        # but this is probably quite inefficient. do better?
        if not self.blt_config.teacher_force_boundaries:
            patch_lens = torch.ones_like(byte_input_ids, dtype=torch.long, device=input_ids.device)

        # Replace generation config with any overrides.
        generation_config = self._generation_config.replace(**generation_kwargs)
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        byte_input_ids = move_to_device(byte_input_ids, self.device)
        batch_size, prompt_len = byte_input_ids.shape
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Output containers
        generated = byte_input_ids
        all_logits: Optional[List[torch.Tensor]] = [] if return_logits else None
        all_logprobs: Optional[List[torch.Tensor]] = [] if return_logprobs else None


        # Timing stats
        time_to_first_token = None
        decode_start_time = None
        setup_time = None
        tokens_generated = 0

        # TODO(benjaminm): should probably get rid of BYTE_EXPANSION_FACTOR and handle length diff on the caller side
        if generation_config.max_new_tokens is not None:
            max_length = prompt_len + generation_config.max_new_tokens * BYTE_EXPANSION_FACTOR
        elif generation_config.max_length is not None:
            max_length = generation_config.max_length * BYTE_EXPANSION_FACTOR
        else:
            max_length = None  # Generate until EOS or stop tokens or OOM...

        prefill_cache_leftpad = None
        if generation_config.use_cache:
            if attention_mask is None:
                attention_mask = torch.ones_like(byte_input_ids, dtype=torch.bool, device=self.device)
            prefill_cache_leftpad = attention_mask_to_cache_leftpad(attention_mask).to(self.device)

        # Initialize/Reset the inference cache
        if generation_config.use_cache:
            if max_length is None:
                raise OLMoConfigurationError(
                    "max_length or max_new_tokens must be provided if use_cache is True"
                )
            self.prepare_inference_cache(batch_size, max_length)
        else:
            self.free_inference_cache()

        # BLT divergence: allow .until and handle stop token sequences
        stop_token_sequences = []

        if generation_config.until is not None:
            stop_token_sequences += [
                torch.tensor(self.tokenizer.encode(x), device=self.model.device, dtype=torch.long)
                for x in generation_config.until
            ]

        if generation_config.stop_token_ids is not None:
            stop_token_sequences += [
                torch.tensor([x], device=self.model.device, dtype=torch.long)
                for x in generation_config.stop_token_ids
            ]

        pbar = tqdm(
            desc="Generating tokens",
            unit="tokens",
            total=(max_length - prompt_len) if max_length is not None else None,
            disable=True, # TEMP DEBUG
            miniters=10,
            colour="blue",
        )
        forward_start_time = None
        last_token_is_boundary = False
        is_first_forward = True
        while not ((max_length is not None and generated.shape[1] >= max_length) or finished.all()):
            input_ids_for_model = (
                generated
                if (is_first_forward or not generation_config.use_cache)
                else generated[:, -1:]
            )
            cache_leftpad = (
                prefill_cache_leftpad if is_first_forward and generation_config.use_cache else None
            )
            patch_lens_for_model = torch.tensor(
                [[1]],
                device=patch_lens.device,
                dtype=patch_lens.dtype
            ) if not is_first_forward and generation_config.use_cache else patch_lens

            forward_start_time = time.perf_counter()
            next_token_logits = self.model.inference_forward(  # type: ignore
                input_ids_for_model,
                patch_lens=patch_lens_for_model,
                last_token_is_boundary=last_token_is_boundary,
                logits_to_keep=1,
                blt_config=self.blt_config,
                cache_leftpad=cache_leftpad if generation_config.use_cache else None,
            )
            is_first_forward = False

            next_tokens = select_next_token(
                next_token_logits.squeeze(1),
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            if next_tokens[0].item() == self.model.end_of_subword_token_blt:
                # possible but two consecutive boundaries are never seen during training
                # do we need to handle this case / forbid consecutive boundaries?
                assert not last_token_is_boundary
                last_token_is_boundary = True
            else:
                if last_token_is_boundary:
                    # start of a new patch
                    patch_lens = torch.cat(
                        [patch_lens, torch.tensor([[1]], device=patch_lens.device, dtype=patch_lens.dtype)],
                        dim=1,
                    )
                else:
                    # patch gets longer
                    patch_lens[:, -1] += 1

                last_token_is_boundary = False

                if all_logits is not None:
                    all_logits.append(next_token_logits)
                if all_logprobs is not None:
                    all_logprobs.append(
                        selective_log_softmax(next_token_logits, next_tokens.unsqueeze(-1))
                    )

                # Force EOS for (previously) finished sequences
                next_tokens = torch.where(finished, torch.full_like(next_tokens, eos), next_tokens)

                # Handle finished sequences
                stop_hit = next_tokens.eq(eos)
                # Also check for stop tokens if provided
                for stop_sequence in stop_token_sequences:
                    stop_hit |= (generated[:, -len(stop_sequence):] == stop_sequence).all(dim=1)

                finished |= stop_hit

                # Append next tokens
                generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)

                if log_timing and tokens_generated == 0:
                    torch.cuda.synchronize()
                    decode_start_time = time.perf_counter()
                    time_to_first_token = decode_start_time - forward_start_time
                    setup_time = forward_start_time - start_time
                tokens_generated += 1
                pbar.update(1)

                if stream:
                    RED = "\033[0;31m"
                    GREEN = "\033[0;32m"
                    RESET = "\033[0;0m"

                    if tokens_generated != 1:
                        tokens_to_print = generated[0][:-1].tolist() if tokens_generated == 1 else generated[0][-2:-1].tolist()

                        if last_token_is_boundary and tokens_generated > 1:
                            print(RED + self.tokenizer.decode(tokens_to_print) + RESET, end="", flush=True)
                        else:
                            if tokens_generated > 1:
                                print(GREEN + self.tokenizer.decode(tokens_to_print) + RESET, end="", flush=True)
                            else:
                                print(self.tokenizer.decode(tokens_to_print), end="", flush=True)

        pbar.close()

        if stream:
            RED = "\033[0;31m"
            GREEN = "\033[0;32m"
            RESET = "\033[0;0m"
            print(RED + self.tokenizer.decode(generated[0][-1:].tolist()) + RESET, end="", flush=True)
            print()

        logits = logprobs = None
        if return_logits and all_logits:
            logits = torch.cat(all_logits, dim=1)

            # check if greedy
            self.free_inference_cache()
            with torch.no_grad():
                reference_out, _ = self.model.student_forward(  # type: ignore
                    input_ids=generated,
                    labels=get_labels({"input_ids": generated}),
                    patch_lens=patch_lens,
                    blt_config=self.blt_config,
                )
            # last logit is potentially wrong since last_token_is_boundary=True in reference, so skip it
            assert (torch.argmax(logits[:, -tokens_generated-1:], -1) == reference_out.logits[:, -tokens_generated-1:-1].argmax(-1)).all()
            assert torch.allclose(logits[:, -tokens_generated-1:], reference_out.logits[:, -tokens_generated-1:-1], rtol=1e-1, atol=1)
        if return_logprobs and all_logprobs:
            logprobs = torch.cat(all_logprobs, dim=1)

        if log_timing:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time = end_time - start_time
            total_tokens = generated.numel()
            prefill_tokens = prompt_len * batch_size
            completion_tokens = tokens_generated * batch_size
            tokens_per_sec_total = total_tokens / total_time
            tokens_per_sec_per_seq = tokens_generated / total_time
            pad_count = (generated == pad).sum().item()
            pad_percentage = (pad_count / total_tokens) * 100 if total_tokens > 0 else 0.0

            stats_lines = [
                f"\n{'=' * 60}",
                "GENERATION STATISTICS",
                f"  Batch size: {batch_size:,} | Prompt len: {prompt_len:,} tokens",
                f"  Tokens generated: {tokens_generated:,} per sequence | Total: {total_tokens:,}",
                f"  Seq length: {prompt_len:,} → {prompt_len + tokens_generated:,}",
                f"  Padding stats: {pad_count:,} / {total_tokens:,} ({pad_percentage:.1f}%)",
            ]
            if decode_start_time and forward_start_time and time_to_first_token:
                decode_time = end_time - decode_start_time
                completion_time = end_time - decode_start_time
                stats_lines.append(
                    f"  Throughput:\n"
                    f"    Setup: {setup_time:.3f}s | Prefill: {time_to_first_token:.3f}s | Decode: {decode_time:.3f}s | Total: {total_time:.3f}s\n"
                    f"    Overall TPS: {tokens_per_sec_per_seq:.1f} /seq | {tokens_per_sec_total:.1f} /total\n"
                    f"    Prefill TPS: {prompt_len / time_to_first_token:.1f} /seq | {prefill_tokens / time_to_first_token:.1f} /total\n"
                    f"    Completion TPS: {tokens_generated / completion_time:.1f} /seq | {completion_tokens / completion_time:.1f} /total",
                )

            stats_lines.append(f"{'=' * 60}")
            log_or_print(log, "\n".join(stats_lines))

        if completions_only:
            generated = generated[:, prompt_len:]

        # convert to token-level ids / logits / logprobs
        # decoding errors could be a problem here - but not really a way around
        generated_text = self.tokenizer.decode(generated[0].tolist())
        generated_subword_ids = torch.tensor([self.tokenizer.hf_tokenizer.encode(generated_text)], dtype=torch.int64, device=self.device)
        _, patch_lens = self.tokenizer.get_tokens_and_patch_lengths(generated_subword_ids[0].tolist(), add_bos=False)
        patch_lens = torch.tensor([patch_lens], dtype=torch.int32, device=self.device)
        patch_ids = blt_utils.lengths_to_ids(patch_lens, generated.shape[-1] - 1).to(self.device)

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
        blt_config: Optional[BLTConfig] = None,
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
                if blt_config is None:
                    blt_config = BLTConfig.from_dict(config_dict["train_module"]["blt_config"])
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

        log_or_print(log, f"{transformer_config}")
        log_or_print(log, f"{generation_config}")
        log_or_print(log, f"{blt_config}")
        model = transformer_config.build()

        # DEBUG flash attention needs bf16
        for block in model.blocks.values():
            block.to(torch.bfloat16)

        generation_module = cls(
            model,
            cast(ByteTokenizerConfig, tokenizer_config).build(),
            cast(BLTConfig, blt_config),
            generation_config,
            **kwargs
        )

        # Load checkpoint
        generation_module.load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            process_group=process_group,
            work_dir=work_dir,
            pre_download=pre_download,
            load_thread_count=load_thread_count,
        )

        return generation_module