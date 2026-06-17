import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
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
    broadcast_object,
    get_fs_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.generate.generation_module import GenerationConfig, GenerationModule
from olmo_core.generate.sampling import select_next_token
from olmo_core.generate.utils import selective_log_softmax
from olmo_core.io import is_url, join_path, normalize_path
from olmo_core.nn.attention import (
    Attention,
    AttentionBackendName,
    FastCompressiveLandmarkAttention,
    FastLandmarkAttention,
    SparseLandmarkAttention,
)
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import gc_cuda, get_default_device, log_or_print, move_to_device

log = logging.getLogger(__name__)

# Landmark attention variants that need prompt-side landmark insertion + "one long local block"
# decoding during generation.
_LANDMARK_ATTENTION_TYPES = (FastLandmarkAttention, SparseLandmarkAttention)


def _insert_landmark_tokens(input_ids: torch.Tensor, mem_freq: int, mem_id: int) -> torch.Tensor:
    """
    Insert a landmark token after every ``mem_freq`` content tokens of each row, reproducing the
    block structure that landmark models are trained on (a landmark at every position ``p`` with
    ``(p + 1) % (mem_freq + 1) == 0``). A trailing partial block (fewer than ``mem_freq`` tokens) is
    left without a landmark.

    :param input_ids: Content token IDs of shape ``(batch_size, seq_len)`` (no landmarks).
    :param mem_freq: Number of content tokens between landmarks (landmark block size is
        ``mem_freq + 1``).
    :param mem_id: The landmark token ID to insert.

    :returns: Token IDs of shape ``(batch_size, seq_len + seq_len // mem_freq)`` with landmarks
        inserted at the fixed periodic positions.
    """
    B, C = input_ids.shape
    n_land = C // mem_freq
    if n_land == 0:
        return input_ids
    block_size = mem_freq + 1
    out_len = C + n_land
    out = input_ids.new_full((B, out_len), mem_id)
    pos = torch.arange(out_len, device=input_ids.device)
    content_pos = pos[((pos + 1) % block_size) != 0]  # everything except the landmark slots
    out[:, content_pos] = input_ids
    return out


def _build_landmark_prompt(
    input_ids: torch.Tensor, mem_freq: int, mem_id: int, *, mode: str, pad_id: int
) -> torch.Tensor:
    """
    Build the landmark-structured prompt fed to prefill from a content-only prompt.

    In ``"generation_only"`` mode the final partial block is padded with ``pad_id`` up to the next
    landmark position, so the prompt always ends with a landmark token (keeping landmarks at the
    trained periodic positions). In ``"extend_last_block"`` mode a trailing partial block is left
    as-is (it stays part of the growing local block during decode).
    """
    content = input_ids
    if mode == "generation_only":
        pad_len = (-content.shape[1]) % mem_freq
        if pad_len:
            pad_block = content.new_full((content.shape[0], pad_len), pad_id)
            content = torch.cat([content, pad_block], dim=1)
    return _insert_landmark_tokens(content, mem_freq, mem_id)


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
        # Attention blocks get a KV cache; recurrent mixers (e.g. GatedDeltaNet) get a
        # conv+recurrent state cache via init_state_cache.
        for block in self.model.blocks.values():
            mixer = block.attention
            if isinstance(mixer, Attention):
                if mixer.kv_cache_manager is None:
                    mixer.init_kv_cache_manager(batch_size, max_seq_len)
                else:
                    mixer.kv_cache_manager.reset(batch_size, max_seq_len)
            elif hasattr(mixer, "init_state_cache"):
                mixer.init_state_cache(batch_size, max_seq_len)  # type: ignore[attr-defined]

    def free_inference_cache(self):
        for block in self.model.blocks.values():
            mixer = block.attention
            if isinstance(mixer, Attention):
                mixer.kv_cache_manager = None
            elif hasattr(mixer, "state_cache"):
                mixer.state_cache = None  # type: ignore[attr-defined]

    def _non_kv_cache_mixer_types(self) -> Set[str]:
        """
        Return the type names of any sequence mixers in the model that support neither a KV cache
        (:class:`~olmo_core.nn.attention.Attention`) nor a recurrent state cache (mixers exposing
        ``init_state_cache``, e.g. :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet`).

        Such a mixer keeps no decode state, so step-by-step decoding with ``use_cache=True`` would
        feed it only the latest token while silently dropping all prior context. Used to guard
        against that wrong code path. Empty for attention and recurrent-with-cache models.
        """
        return {
            type(block.attention).__name__
            for block in self.model.blocks.values()
            if not isinstance(block.attention, Attention)
            and not hasattr(block.attention, "init_state_cache")
        }

    def _landmark_attention_layers(self) -> List[Attention]:
        """Return the model's landmark attention layers (empty if this is not a landmark model)."""
        layers: List[Attention] = []
        for block in self.model.blocks.values():
            attn = block.attention
            if isinstance(attn, _LANDMARK_ATTENTION_TYPES):
                layers.append(cast(Attention, attn))
        return layers

    def _set_landmark_eval_decode(
        self,
        prompt_len: int,
        mode: str,
        top_k: Optional[int] = None,
        nonselected_landmark_mass: Optional[float] = None,
    ):
        for attn in self._landmark_attention_layers():
            if isinstance(attn, FastCompressiveLandmarkAttention):
                attn.set_landmark_eval_decode(
                    prompt_len,
                    mode,
                    top_k=top_k,
                    nonselected_landmark_mass=nonselected_landmark_mass,
                )
            else:
                attn.set_landmark_eval_decode(prompt_len, mode, top_k=top_k)  # type: ignore[attr-defined]

    def _clear_landmark_eval_decode(self):
        for attn in self._landmark_attention_layers():
            attn.clear_landmark_eval_decode()  # type: ignore[attr-defined]

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

        # Guard the no-decode-state wrong path: a sequence mixer that supports neither a KV cache
        # nor a recurrent state cache carries no per-step state, so cached decoding would feed it
        # only the latest token and silently drop all prompt context. Attention (KV cache) and
        # GatedDeltaNet (conv+recurrent state cache) are both fine; this only fires for a mixer that
        # supports neither. Fail loudly rather than decode garbage.
        if generation_config.use_cache:
            non_kv_cache_mixers = self._non_kv_cache_mixer_types()
            if non_kv_cache_mixers:
                raise OLMoConfigurationError(
                    "use_cache=True is not supported for models with sequence mixers "
                    f"{sorted(non_kv_cache_mixers)} that have no decode-time cache (neither a KV "
                    "cache nor a recurrent state cache). Set use_cache=False to generate correctly "
                    "(slower), or add state caching for these layers."
                )

        input_ids = move_to_device(input_ids, self.device)

        # Landmark attention: insert landmark tokens into the *prompt* every ``mem_freq`` content
        # tokens (so prefill sees the trained block structure) and decode plain content tokens as
        # "one long local block". This keeps the eval harness landmark-agnostic -- it only ever sees
        # content tokens in and content tokens out.
        landmark_layers = self._landmark_attention_layers()
        landmark_active = len(landmark_layers) > 0
        orig_input_ids = input_ids
        if landmark_active:
            if generation_config.landmark_mem_id is None:
                raise OLMoConfigurationError(
                    "This is a landmark-attention model; set GenerationConfig.landmark_mem_id "
                    "(the landmark token ID) to generate."
                )
            if not generation_config.use_cache:
                raise OLMoConfigurationError(
                    "Landmark-attention generation requires use_cache=True."
                )
            if attention_mask is not None and not bool(attention_mask.all()):
                raise OLMoConfigurationError(
                    "Landmark-attention generation does not support left-padding / attention masks "
                    "(blocks are tied to absolute position; use batch_size=1)."
                )
            mem_freqs = {int(getattr(a, "mem_freq")) for a in landmark_layers}
            if len(mem_freqs) != 1:
                raise OLMoConfigurationError(
                    f"Landmark layers have inconsistent mem_freq values: {sorted(mem_freqs)}"
                )
            pad_id = generation_config.landmark_pad_id
            if pad_id is None:
                pad_id = generation_config.pad_token_id
            input_ids = _build_landmark_prompt(
                input_ids,
                mem_freqs.pop(),
                generation_config.landmark_mem_id,
                mode=generation_config.landmark_decode_mode,
                pad_id=pad_id,
            )

        batch_size, prompt_len = input_ids.shape
        if landmark_active:
            self._set_landmark_eval_decode(
                prompt_len,
                generation_config.landmark_decode_mode,
                top_k=generation_config.landmark_top_k_blocks,
                nonselected_landmark_mass=generation_config.landmark_nonselected_mass,
            )
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
            # ``max_length`` is an absolute cap on ``generated.shape[1]``. Eval harnesses set it
            # to ``len(content_prompt) + max_gen_toks``, unaware that landmark generation inserts
            # memory tokens into the prompt. Those inserted tokens count toward
            # ``generated.shape[1]``, so without compensation they eat into (and can zero out) the
            # new-token budget. Extend the cap by the number of inserted landmark tokens so the
            # content-token budget stays ``max_gen_toks``.
            if landmark_active:
                max_length += prompt_len - orig_input_ids.shape[1]
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

        # Landmark hard top-k retrieval is applied only on single-query decode steps, never during the
        # batched prefill. The first generated token's logits come from the *last prompt token's*
        # query, so if prefill produced it that token would escape top-k entirely. Instead, prefill all
        # but the final prompt token here and let the decode loop below start from that final token --
        # then retrieval gates the first generated token too. (Requires a KV cache and >= 2 prompt
        # tokens; landmark generation already requires use_cache and batch_size=1 / no left-padding.)
        landmark_decode_first_token = (
            landmark_active and generation_config.use_cache and prompt_len >= 2
        )
        if landmark_decode_first_token:
            self.model(input_ids[:, :-1], logits_to_keep=1, cache_leftpad=prefill_cache_leftpad)

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
            if landmark_decode_first_token:
                # Prompt[:-1] is already cached above; every step is a single-token decode, starting
                # from the final prompt token (whose query yields the first generated token).
                input_ids_for_model = generated[:, -1:]
                cache_leftpad = None
            else:
                is_first_forward = generated.shape[1] == prompt_len
                input_ids_for_model = (
                    generated
                    if (is_first_forward or not generation_config.use_cache)
                    else generated[:, -1:]
                )
                cache_leftpad = (
                    prefill_cache_leftpad
                    if is_first_forward and generation_config.use_cache
                    else None
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

        if landmark_active:
            # ``generated`` holds the landmark-inserted prompt followed by plain content tokens; the
            # caller never asked for landmarks, so report the prompt in its original content space.
            self._clear_landmark_eval_decode()
            completion = generated[:, prompt_len:]
            generated = (
                completion if completions_only else torch.cat([orig_input_ids, completion], dim=1)
            )
        elif completions_only:
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

        train_module_dir = broadcast_object(train_module_dir)
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
        attention_backend: Optional[AttentionBackendName] = None,
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
            attention_backend: If provided, override the config to use this attention backend.
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
        checkpoint_landmark_mem_id: Optional[int] = None
        if transformer_config is None and get_rank(process_group) == 0:
            config_path = join_path(checkpoint_dir, "config.json")
            with cached_path(config_path).open() as f:
                config_dict = json.load(f)
            try:
                # Avoid loading the entire experiment config b/c we don't care about validation outside
                # of the transformer config and the tokenizer config
                transformer_config = TransformerConfig.from_dict(config_dict["model"])
            except KeyError as e:
                raise OLMoConfigurationError(
                    f"Failed to load config from checkpoint at {config_path}: missing required field {e}"
                ) from e

            # The tokenizer config is only needed to synthesize a generation config when one
            # isn't provided. Tolerate checkpoints whose config stores the tokenizer elsewhere
            # or stores `dataset` in a different shape (e.g. composable data configs where
            # `dataset` is a list of sources), so loading still works when a generation config
            # is passed explicitly.
            try:
                tokenizer_config = TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])
            except (KeyError, TypeError):
                tokenizer_config = None

            # For composable data pipelines the dataset field is a list of InstanceSourceConfig
            # dicts; a LandmarkInstanceSourceConfig carries `mem_id` at the top level.
            dataset_cfg = config_dict.get("dataset")
            if isinstance(dataset_cfg, list):
                for src in dataset_cfg:
                    if isinstance(src, dict) and "mem_id" in src:
                        checkpoint_landmark_mem_id = int(src["mem_id"])
                        break

        # Create work directory on rank 0
        work_dir = Path(
            work_dir or (tempfile.mkdtemp() if get_rank(process_group) == 0 else "/tmp")
        )

        # Broadcast config and work_dir to all ranks
        (
            transformer_config,
            work_dir,
            tokenizer_config,
            checkpoint_landmark_mem_id,
        ) = broadcast_object(
            (transformer_config, work_dir, tokenizer_config, checkpoint_landmark_mem_id)
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
                landmark_mem_id=checkpoint_landmark_mem_id,
            )
            log_or_print(
                log,
                f"No generation config provided, using defaults from checkpoint config: {generation_config}",
            )

        # Auto-fill landmark_mem_id from checkpoint config when the caller didn't set it.
        if checkpoint_landmark_mem_id is not None and generation_config.landmark_mem_id is None:
            import dataclasses

            generation_config = dataclasses.replace(
                generation_config, landmark_mem_id=checkpoint_landmark_mem_id
            )
            log_or_print(
                log,
                f"Auto-filled landmark_mem_id={checkpoint_landmark_mem_id} from checkpoint config",
            )

        # Build model and generation module
        if dtype is not None:
            dtype = DType(dtype)
            transformer_config.apply(
                lambda c: setattr(c, "dtype", dtype) if hasattr(c, "dtype") else None
            )

        if attention_backend is not None:
            attention_backend.assert_supported()

            def set_attention_backend(c):
                mixer = getattr(c, "sequence_mixer", None) or getattr(c, "attention", None)
                if mixer is None and hasattr(c, "backend"):
                    mixer = c
                if mixer is not None and hasattr(mixer, "backend"):
                    setattr(mixer, "backend", attention_backend)

            transformer_config.apply(set_attention_backend)

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

    @classmethod
    @torch.no_grad()
    def from_checkpoints(
        cls,
        checkpoint_dirs: List[PathOrStr],
        dtype: Optional[DType] = None,
        **kwargs,
    ) -> "TransformerGenerationModule":
        if len(checkpoint_dirs) == 1:
            return cls.from_checkpoint(checkpoint_dirs[0], dtype=dtype, **kwargs)

        log.info(f"Merging {len(checkpoint_dirs)} checkpoints")

        # Override device to CPU for intermediate checkpoint loading to save GPU memory
        cpu_kwargs = kwargs.copy()
        cpu_kwargs["device"] = torch.device("cpu")

        log.info(
            "Loading checkpoints on CPU for merging, will transfer to target device at the end"
        )
        log.info(f"Merging {checkpoint_dirs[0]}")
        first_generation_module = cls.from_checkpoint(
            checkpoint_dirs[0], dtype=DType.float32, **cpu_kwargs
        )

        # Get the state dict from the first module to use as accumulator
        merged_state_dict = first_generation_module.state_dict()

        # Free the first module since we have its state dict
        del first_generation_module
        gc_cuda()

        # Average weights from all checkpoints
        for i, checkpoint_dir in enumerate(checkpoint_dirs[1:], start=2):
            log.info(f"Merging {checkpoint_dir}")
            next_generation_module = cls.from_checkpoint(
                checkpoint_dir, dtype=DType.float32, **cpu_kwargs
            )
            next_state_dict = next_generation_module.state_dict()
            del next_generation_module

            # Average the weights
            for key in merged_state_dict["model"].keys():
                target_tensor = merged_state_dict["model"][key]
                if torch.is_tensor(target_tensor) and torch.is_floating_point(target_tensor):
                    source_tensor = next_state_dict["model"].pop(key)
                    assert target_tensor.shape == source_tensor.shape
                    # in-place operations for better memory consumption
                    target_tensor.mul_((i - 1) / i).add_(source_tensor, alpha=1.0 / i)
                    del target_tensor
                    del source_tensor

            # Free memory from the temporary module and run garbage collection
            del next_state_dict
            gc_cuda()

        # Now load the final model on the target device with the correct dtype
        if dtype == DType.float32:
            # If target dtype is float32, we can reuse the merged state dict
            log.info("Loading merged checkpoint with dtype float32")
            final_generation_module = cls.from_checkpoint(
                checkpoint_dirs[0], dtype=DType.float32, **kwargs
            )
            final_generation_module.load_state_dict(merged_state_dict)
            return final_generation_module

        # Otherwise, load with the target dtype and convert the merged weights
        log.info(f"Loading merged checkpoint with dtype {dtype}")
        final_generation_module = cls.from_checkpoint(checkpoint_dirs[0], dtype=dtype, **kwargs)
        final_state_dict = final_generation_module.state_dict()
        assert merged_state_dict["model"].keys() == final_state_dict["model"].keys()

        # Convert merged state dict to the target dtype
        for key in merged_state_dict["model"].keys():
            merged_state_dict_tensor = merged_state_dict["model"][key]
            if not torch.is_tensor(merged_state_dict_tensor):
                continue
            final_state_dict_tensor = final_state_dict["model"][key]
            if not torch.is_tensor(final_state_dict_tensor):
                continue
            if merged_state_dict_tensor.dtype != final_state_dict_tensor.dtype:
                merged_state_dict["model"][key] = merged_state_dict_tensor.to(
                    final_state_dict_tensor.dtype
                )

        final_generation_module.load_state_dict(merged_state_dict)

        gc_cuda()

        return final_generation_module
