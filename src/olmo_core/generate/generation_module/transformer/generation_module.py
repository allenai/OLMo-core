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
from olmo_core.nn.blt import utils as blt_utils
import olmo_core.nn.blt.utils as blt_utils
from olmo_core.nn.mamba import Mamba
from olmo_core.nn.xlstm import XLSTM
from olmo_core.nn.transformer import Transformer, TransformerConfig, TransformerType
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, log_or_print, move_to_device

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
        return_timing: bool = False,
        completions_only: bool = False,
        log_timing: bool = True,
        profile: bool = False,
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

        if profile:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/transformer_generation_module'),
            )
            prof.__enter__()
        else:
            prof = None

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

            if prof is not None:
                prof.step()

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

        if prof is not None:
            prof.__exit__(None, None, None)

        logits = logprobs = None
        if return_logits and all_logits:
            logits = torch.cat(all_logits, dim=1)
        if return_logprobs and all_logprobs:
            logprobs = torch.cat(all_logprobs, dim=1)

        timings = {}

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
                stats_lines.append(
                    f"  Throughput:\n"
                    f"    Setup: {setup_time:.3f}s | Prefill: {time_to_first_token:.3f}s | Decode: {decode_time:.3f}s | Total: {total_time:.3f}s\n"
                    f"    Overall TPS: {tokens_per_sec_per_seq:.1f} /seq | {tokens_per_sec_total:.1f} /total\n"
                    f"    Prefill TPS: {prompt_len / time_to_first_token:.1f} /seq | {prefill_tokens / time_to_first_token:.1f} /total\n"
                    f"    Completion TPS: {tokens_generated / decode_time:.1f} /seq | {completion_tokens / decode_time:.1f} /total",
                )

                timings["setup_time"] = setup_time
                timings["decode_time"] = decode_time
                timings["tokens_generated"] = completion_tokens
                timings["ttft"] = time_to_first_token
                timings["tpot"] = decode_time / completion_tokens if completion_tokens > 0 else None

            stats_lines.append(f"{'=' * 60}")
            log_or_print(log, "\n".join(stats_lines))

        if completions_only:
            generated = generated[:, prompt_len:]
            # NOTE: completions_only does not apply to logits/logprobs. They are already computed only for completions.
        
        if return_timing:
            return generated, logits, logprobs, timings
        else:
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

        # DEBUG force enable flash attention
        transformer_config.block.attention.use_flash = True

        log_or_print(log, f"{transformer_config}")
        log_or_print(log, f"{generation_config}")
        model = transformer_config.build()

        # DEBUG flash attention needs bf16
        for block in model.blocks.values():
            block.to(torch.bfloat16)
        
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
            elif hasattr(block, "xlstm") and isinstance(block.xlstm, XLSTM):
                xlstm = block.xlstm
                if xlstm.xlstm_cache_manager is None:
                    xlstm.init_xlstm_cache_manager(batch_size)
                else:
                    xlstm.xlstm_cache_manager.reset(batch_size)

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
    def model_forward(
        self,
        input_ids: torch.Tensor,
        pad_to_multiple_of: int = 128,
        n_extra_global_tokens: int = 32, # probably needs a better solution than a fixed extra budget
        **kwargs
    ):
        self._set_model_mode("eval")

        original_length = input_ids.shape[1]
        if original_length % pad_to_multiple_of != 0:
            new_length = math.ceil(original_length / pad_to_multiple_of) * pad_to_multiple_of
            pad_length = new_length - original_length
            input_ids = F.pad(input_ids, (0, pad_length), value=self.tokenizer.pad_token_id)

        # compute patch lengths
        # if teacher_force_boundaries=False, only patch_lens.shape[1] is used as the budget of patches
        patch_lens = []
        for example_idx in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[example_idx].tolist())
            subword_tokens = self.tokenizer.hf_tokenizer.encode(text)
            _, example_patch_lens = self.tokenizer.get_tokens_and_patch_lengths(subword_tokens, add_bos=True)
            patch_lens.append(example_patch_lens)

        pad_to = max(len(x) for x in patch_lens) + n_extra_global_tokens
        for example_idx in range(input_ids.shape[0]):
            while len(patch_lens[example_idx]) < pad_to:
                patch_lens[example_idx].append(0)

        patch_lens = torch.tensor(patch_lens, dtype=torch.long, device=input_ids.device)
        input_ids = move_to_device(input_ids, self.device)
        patch_lens = move_to_device(patch_lens, self.device)

        labels = F.pad(input_ids[:, 1:], (0, 1), value=-100)

        return self.model.student_forward(  # type: ignore
            input_ids,
            labels=labels,
            patch_lens=patch_lens,
            blt_config=self.blt_config,
            **kwargs
        )[0].logits[:, :original_length, :]

    def _get_last_non_boundary_token(self, input_ids: torch.Tensor):
        input_ids = input_ids.flip(1)
        first_non_boundary_idx = (input_ids != self.model.end_of_subword_token_blt).float().argmax(dim=1)  # type: ignore
        return torch.take_along_dim(input_ids, first_non_boundary_idx.unsqueeze(1), 1)

    @torch.inference_mode()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        return_logprobs: bool = True,
        return_timing: bool = False,
        completions_only: bool = False,
        log_timing: bool = True,
        stream: bool = False,
        verify: bool = False,
        force_boundary_every: Optional[int] = None,
        max_patch_length_decode: Optional[int] = 128, # 128 is longest dolma2 tokenizer token
        profile: bool = False,
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
        self._set_model_mode("eval")
        start_time = time.perf_counter()

        expand_input_ids = self.model.local_encoder.add_expanded_embeddings  # type: ignore

        # convert to byte ids and compute patch lengths
        byte_input_ids = []
        expanded_input_ids = []
        patch_lens = []

        for example_idx in range(input_ids.shape[0]):
            current_byte_input_ids, current_patch_lens = self.tokenizer.get_tokens_and_patch_lengths(
                input_ids[example_idx].tolist(),
                add_bos=True,
                strip_pad=True
            )
            byte_input_ids.append(torch.tensor(current_byte_input_ids, dtype=torch.long))
            if expand_input_ids:
                expanded_input_ids.append(torch.tensor(self.tokenizer.expand_byte_ids(current_byte_input_ids), dtype=torch.long))
            patch_lens.append(torch.tensor(current_patch_lens, dtype=torch.long))

        byte_input_ids = blt_utils.pad_left(byte_input_ids, value=self.tokenizer.pad_token_id, multiple_of=1)
        patch_lens = blt_utils.pad_left(patch_lens, value=1, multiple_of=1)

        byte_input_ids = move_to_device(byte_input_ids, self.device)
        patch_lens = move_to_device(patch_lens, self.device)
        
        if expanded_input_ids:
            expanded_input_ids = blt_utils.pad_left(expanded_input_ids, value=self.tokenizer.pad_token_id, multiple_of=1)
            expanded_input_ids = move_to_device(expanded_input_ids, self.device)
        else:
            expanded_input_ids = None

        sequence_start_indices = (byte_input_ids == self.tokenizer.pad_token_id).sum(-1)
        batch_size, prompt_len = byte_input_ids.shape
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Replace generation config with any overrides.
        generation_config = self._generation_config.replace(**generation_kwargs)
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        # max length *in tokens*, not bytes
        if generation_config.max_new_tokens is not None:
            # input_ids.shape[1] is not accurate but we don't yet know how many prefill tokens
            max_length = input_ids.shape[1] + generation_config.max_new_tokens
        elif generation_config.max_length is not None:
            max_length = generation_config.max_length
        else:
            max_length = None  # Generate until EOS or stop tokens or OOM...

        # Initialize/Reset the inference cache
        if generation_config.use_cache:
            if max_length is None:
                raise OLMoConfigurationError(
                    "max_length or max_new_tokens must be provided if use_cache is True"
                )
            self.prepare_inference_cache(batch_size, max_length)
        else:
            self.free_inference_cache()

        boundary_mask, cached_encoder_outputs = self.model.prefill_boundary_prediction_forward(  # type: ignore
            byte_input_ids,
            patch_lens=patch_lens,
            expanded_input_ids=expanded_input_ids,
            sequence_start_indices=sequence_start_indices,
            blt_config=self.blt_config,
        )

        # Output containers
        generated = byte_input_ids

        # Timing stats
        time_to_first_token = None
        decode_start_time = None
        setup_time = None
        max_n_prefill_patches = boundary_mask.sum(-1).max().item()
        tokens_generated_plus_prefilled = max_n_prefill_patches
        bytes_generated = 0

        prefill_cache_leftpad = None
        if generation_config.use_cache:
            n_boundaries = boundary_mask.sum(-1)
            attention_mask: torch.Tensor = (
                torch.arange(max_n_prefill_patches, device=byte_input_ids.device).unsqueeze(0).repeat(batch_size, 1)
                < n_boundaries[:, None]
            )
            attention_mask = attention_mask.flip(1)
            prefill_cache_leftpad = attention_mask_to_cache_leftpad(attention_mask).to(self.device)

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
            disable=not log_timing,
            miniters=10,
            colour="blue",
        )
        forward_start_time = None
        boundary_state = blt_utils.MaskState(torch.zeros(batch_size, dtype=torch.bool, device=self.device))
        next_tokens = torch.full((input_ids.shape[0],), self.model.end_of_subword_token_blt, device=self.device, dtype=torch.long)  # type: ignore
        non_boundary_next_tokens = byte_input_ids[:, -1:].clone()

        bytes_generated_at_last_boundary = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        is_first_forward = True

        if profile:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                with_flops=True,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/blt_transformer_generation_module'),
            )
            prof.__enter__()
        else:
            prof = None

        while not ((max_length is not None and tokens_generated_plus_prefilled >= max_length) or finished.all()):
            input_ids_for_model = (
                generated
                if (is_first_forward or not generation_config.use_cache)
                else non_boundary_next_tokens
            )
            if expand_input_ids:
                expanded_input_ids_for_model = torch.zeros_like(input_ids_for_model)
                for i in range(input_ids_for_model.shape[0]):
                    expanded_input_ids_for_model[i, :] = torch.tensor(self.tokenizer.expand_byte_ids(
                        generated[i, :].tolist(),
                        n_last=input_ids_for_model.shape[1],
                    ), device=expanded_input_ids_for_model.device, dtype=expanded_input_ids_for_model.dtype)
            else:
                expanded_input_ids_for_model = None

            cache_leftpad = (
                prefill_cache_leftpad if is_first_forward and generation_config.use_cache else None
            )
            forward_start_time = time.perf_counter()
            next_token_logits = self.model.inference_forward(  # type: ignore
                input_ids_for_model,
                expanded_input_ids=expanded_input_ids_for_model,
                cached_encoder_outputs=cached_encoder_outputs if is_first_forward else None,
                boundary_state=boundary_state,
                sequence_start_indices=sequence_start_indices,
                logits_to_keep=1,
                blt_config=self.blt_config,
                cache_leftpad=cache_leftpad if generation_config.use_cache else None,
            )
            new_next_tokens = select_next_token(
                next_token_logits.squeeze(1),
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            if boundary_state.all():
                if prof is not None:
                    prof.step()
                pbar.update(1)
                tokens_generated_plus_prefilled += 1

                next_tokens = new_next_tokens
                non_boundary_next_tokens = next_tokens.unsqueeze(1).clone() # clone necessary?
            else:
                next_tokens[:] = self.model.end_of_subword_token_blt # type: ignore
                boundary_state.selective_put(new_next_tokens, next_tokens, inv=True)
                boundary_state.selective_put(new_next_tokens, non_boundary_next_tokens[:, -1], inv=True)

            is_first_forward = False

            # for debugging
            if force_boundary_every is not None:
                if (bytes_generated + 1) % force_boundary_every == 0:
                    next_tokens[:] = self.model.end_of_subword_token_blt # type: ignore
                else:
                    # doesn't matter but 36 is whitespace in blt tokenizer
                    next_tokens[:] = 36

            if max_patch_length_decode is not None:
                stop_mask = bytes_generated - bytes_generated_at_last_boundary >= max_patch_length_decode
                next_tokens[stop_mask] = self.model.end_of_subword_token_blt # type: ignore

                if stop_mask.any().item():  # type: ignore
                    log.warning(f"Forcing boundary since patch exceeds length {max_patch_length_decode}.")

            boundary_state = blt_utils.MaskState(next_tokens == self.model.end_of_subword_token_blt)  # type: ignore
            boundary_state.selective_put(bytes_generated, bytes_generated_at_last_boundary)

            # Force EOS for (previously) finished sequences
            next_tokens = torch.where(finished, torch.full_like(next_tokens, eos), next_tokens)

            # Append next tokens
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)

            # Handle finished sequences
            stop_hit = next_tokens.eq(eos)

            # Also check for stop tokens if provided
            for stop_sequence in stop_token_sequences:
                stop_hit |= (generated[:, -len(stop_sequence):] == stop_sequence).all(dim=1)

            finished |= stop_hit

            if log_timing and bytes_generated == 0:
                torch.cuda.synchronize()
                decode_start_time = time.perf_counter()
                time_to_first_token = decode_start_time - forward_start_time
                setup_time = forward_start_time - start_time
            bytes_generated += 1

            if stream:
                RED = "\033[0;31m"
                GREEN = "\033[0;32m"
                RESET = "\033[0;0m"

                if bytes_generated != 1:
                    tokens_to_print = generated[0][:-1].tolist() if bytes_generated == 1 else generated[0][-2:-1].tolist()

                    if boundary_state.mask[0].item() and bytes_generated > 1:
                        print(RED + self.tokenizer.decode(tokens_to_print) + RESET, end="", flush=True)
                    else:
                        if bytes_generated > 1:
                            print(GREEN + self.tokenizer.decode(tokens_to_print) + RESET, end="", flush=True)
                        else:
                            print(self.tokenizer.decode(tokens_to_print), end="", flush=True)

        pbar.close()
        if prof is not None:
            prof.__exit__(None, None, None)

        if stream:
            RED = "\033[0;31m"
            GREEN = "\033[0;32m"
            RESET = "\033[0;0m"
            print(RED + self.tokenizer.decode(generated[0][-1:].tolist()) + RESET, end="", flush=True)
            print()

        # TODO(benjaminm): restore
        # logits = logprobs = None
        # if return_logits and all_logits:
        #     logits = torch.cat(all_logits, dim=1)

        #     # check if greedy
        #     if verify:
        #         self.free_inference_cache()
        #         with torch.no_grad():
        #             reference_out, _ = self.model.student_forward(  # type: ignore
        #                 input_ids=generated,
        #                 labels=get_labels({"input_ids": generated}),
        #                 patch_lens=patch_lens,
        #                 blt_config=self.blt_config,
        #             )
        #         # last logit is potentially wrong since last_token_is_boundary=True in reference, so skip it
        #         assert (torch.argmax(logits[:, -bytes_generated-1:], -1) == reference_out.logits[:, -bytes_generated-1:-1].argmax(-1)).all()
        #         assert torch.allclose(
        #             F.softmax(logits[:, -bytes_generated-1:], -1),
        #             F.softmax(reference_out.logits[:, -bytes_generated-1:-1], -1),
        #             rtol=1e-1,
        #             atol=0.05, # accept max 5% prob diff (highest observed ~1.6%)
        #         )
        # if return_logprobs and all_logprobs:
        #     logprobs = torch.cat(all_logprobs, dim=1)

        timings = {}

        if log_timing:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time = end_time - start_time
            total_tokens = generated.numel()
            prefill_tokens = prompt_len * batch_size
            completion_tokens = bytes_generated * batch_size
            tokens_per_sec_total = total_tokens / total_time
            tokens_per_sec_per_seq = bytes_generated / total_time
            pad_count = (generated == pad).sum().item()
            pad_percentage = (pad_count / total_tokens) * 100 if total_tokens > 0 else 0.0

            stats_lines = [
                f"\n{'=' * 60}",
                "GENERATION STATISTICS",
                f"  Batch size: {batch_size:,} | Prompt len: {prompt_len:,} bytes",
                f"  Bytes generated: {bytes_generated:,} per sequence | Total: {total_tokens:,}",
                f"  Seq length: {prompt_len:,} → {prompt_len + bytes_generated:,}",
                f"  Padding stats: {pad_count:,} / {total_tokens:,} ({pad_percentage:.1f}%)",
            ]
            if decode_start_time and forward_start_time and time_to_first_token:
                decode_time = end_time - decode_start_time
                stats_lines.append(
                    f"  Throughput:\n"
                    f"    Setup: {setup_time:.3f}s | Prefill: {time_to_first_token:.3f}s | Decode: {decode_time:.3f}s | Total: {total_time:.3f}s\n"
                    f"    Overall BPS: {tokens_per_sec_per_seq:.1f} /seq | {tokens_per_sec_total:.1f} /total\n"
                    f"    Prefill BPS: {prompt_len / time_to_first_token:.1f} /seq | {prefill_tokens / time_to_first_token:.1f} /total\n"
                    f"    Completion BPS: {bytes_generated / decode_time:.1f} /seq | {completion_tokens / decode_time:.1f} /total",
                )

                timings["setup_time"] = setup_time
                timings["decode_time"] = decode_time
                timings["bytes_generated"] = completion_tokens
                timings["ttft"] = time_to_first_token
                timings["tpot"] = decode_time / completion_tokens if completion_tokens > 0 else None

            stats_lines.append(f"{'=' * 60}")
            log_or_print(log, "\n".join(stats_lines))

        # convert to token-level ids / logits / logprobs
        # decoding errors could be a problem here - but not really a way around
        generated_subword_ids = []

        for example_idx in range(generated.shape[0]):
            completion_text = self.tokenizer.decode(generated[example_idx, prompt_len:].tolist())
            completion_subword_tokens = self.tokenizer.hf_tokenizer.encode(completion_text)

            if completions_only:
                subword_tokens = completion_subword_tokens
            else:
                subword_tokens = input_ids[example_idx].tolist() + completion_subword_tokens

            generated_subword_ids.append(torch.tensor(subword_tokens, device=input_ids.device, dtype=torch.long))

        generated_subword_ids = blt_utils.pad_right(
            generated_subword_ids,
            value=self.tokenizer.hf_tokenizer.pad_token_id,
            multiple_of=1,
        )

        if return_logits or return_logprobs:
            logits = logprobs = torch.zeros((batch_size, 0)) # not implemented
        else:
            logits = logprobs = None

        if return_timing:
            return generated_subword_ids, logits, logprobs, timings
        else:
            return generated_subword_ids, logits, logprobs

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