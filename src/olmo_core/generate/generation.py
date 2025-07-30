import contextlib
import json
import logging
import tempfile
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from rich import print
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.stateful import Stateful
from tqdm import tqdm

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
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
from olmo_core.generate.config import GenerationConfig
from olmo_core.generate.selection import select_next_token
from olmo_core.generate.utils import selective_log_softmax
from olmo_core.io import is_url, join_path, normalize_path
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, move_to_device

log = logging.getLogger(__name__)


class GenerationModule(Stateful, metaclass=ABCMeta):
    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        """
        Should return the data parallel process group if it's anything other than the default
        process group.
        """
        return None

    def state_dict_to_load(self, metadata: Metadata) -> Dict[str, Any]:
        del metadata
        return self.state_dict()

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load a state dict.
        """
        raise NotImplementedError


class TransformerGenerationModule(GenerationModule):
    """Module for autoregressive text generation with transformer models."""

    def __init__(
        self,
        model: Transformer,
        generation_config: GenerationConfig,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        autocast_precision: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()

        # Build world mesh.
        self.device = device or get_default_device()
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
        self.autocast_precision = autocast_precision
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

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    def model_forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        logits_to_keep: int = 0,
        prefill_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run a forward pass on a micro-batch, returning the logits.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            logits_to_keep: Number of positions to keep from the end (0 = keep all)
            prefill_kv_cache: Whether to prefill the KV cache. If True, the KV cache will be
                prefilled with the input_ids.
            **kwargs: Additional arguments passed to the model
        """
        with self._model_forward_context():
            logits = self.model(  # (batch_size, seq_len, vocab_size)
                input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
                prefill_kv_cache=prefill_kv_cache,
                **kwargs,
            )
            return logits

    @staticmethod
    def _reset_kv_cache(
        model: Transformer, use_cache: bool, batch_size: int, max_seq_len: int, dtype: torch.dtype
    ):
        for block in model.blocks.values():
            if hasattr(block.attention, "reset_kv_cache"):
                block.attention.reset_kv_cache(  # type: ignore
                    use_cache=use_cache, batch_size=batch_size, max_seq_len=max_seq_len, dtype=dtype
                )

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

        # Initialize/Reset the KV cache
        self._reset_kv_cache(
            self.model,
            generation_config.use_cache,
            batch_size,
            generation_config.max_length,
            self.autocast_precision or torch.float32,
        )

        pbar = tqdm(desc="Generating tokens", disable=not log_timing)
        while True:
            token_start_time = time.perf_counter()

            # Determine model inputs based on whether we're using the cache
            is_first_forward = generated.shape[1] == prompt_len
            is_using_cache = generation_config.use_cache and not is_first_forward

            input_ids_for_model = generated[:, -1:] if is_using_cache else generated
            attention_mask_for_model = None if is_using_cache else attention_mask
            logits_to_keep = 1 if (is_using_cache or completions_only) else prompt_len

            logits = self.model_forward(  # (batch_size, generated.shape[1], self.model.vocab_size)
                input_ids_for_model,
                attention_mask=attention_mask_for_model,
                logits_to_keep=logits_to_keep,
                prefill_kv_cache=is_first_forward,
            )
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            if all_logits is not None:
                all_logits.append(logits)  # (batch_size, seq_len, vocab_size)

            # Check if we should stop before generating more tokens
            pbar.update(1)
            if generated.shape[1] >= generation_config.max_length or finished.all():
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

            # Calculate log probabilities for the tokens we just generated.
            if all_logprobs is not None:
                if is_first_forward and prompt_len > 1 and not completions_only:
                    # For the prompt, we already have the ground truth tokens, so we can
                    # calculate their log probabilities all at once from the first forward pass.
                    # But only if we're not in completions_only mode and we have the full prompt logits
                    prompt_log_probs = selective_log_softmax(logits[:, :-1, :], generated[:, 1:])
                    all_logprobs.append(prompt_log_probs)

                # Calculate the log probability of the token that we just selected.
                next_token_log_prob = selective_log_softmax(next_token_logits, next_tokens)
                all_logprobs.append(next_token_log_prob.unsqueeze(1))

            # Handle finished sequences
            # - Check which sequences have generated EOS
            prev_finished = finished.clone()
            just_finished = next_tokens == generation_config.eos_token_id
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
                logprobs = logprobs[:, prompt_len:, :]

        # Erase the KV cache
        self._reset_kv_cache(
            self.model,
            False,
            batch_size,
            generation_config.max_length,
            self.autocast_precision or torch.float32,
        )

        total_time = time.perf_counter() - start_time
        if log_timing:
            log.info("Generation stats:")
            log.info(f"  Batch size: {batch_size}  Prompt length: {prompt_len}")
            log.info(
                f"  Tokens generated: {tokens_generated * batch_size} ({tokens_generated} per sequence, "
                f"{tokens_generated * batch_size / total_time:.1f} tokens/s)"
            )
            log.info(f"  Sequence length extended: {prompt_len} → {prompt_len + tokens_generated}")
            log.info(f"  Total generation time: {total_time:.3f}s")
            if time_to_first_token is not None:
                log.info(f"  Time to first token: {time_to_first_token:.3f}s")
            if token_times:
                avg_inter_token_latency = sum(token_times) / len(token_times)
                log.info(f"  Average inter-token latency: {avg_inter_token_latency * 1000:.1f}ms")

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
