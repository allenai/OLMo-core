from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import torch
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException
from typing_extensions import Self

from .exceptions import OLMoConfigurationError


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


C = TypeVar("C", bound="Config")


@dataclass
class Config:
    """
    A base class for configuration dataclasses.

    .. important::
        When you subclass this you should still decorate your subclasses with
        :func:`@dataclass <dataclasses.dataclass>`. For example::

            @dataclass
            class MyConfig(Config):
                ...

    .. important::
        Config classes need to be serializable, so you should only use simple types for your fields.
        Though you can use nested configs.
    """

    CLASS_NAME_FIELD = "_CLASS_"
    """
    The name of the class name field inject into the dictionary from :meth:`as_dict()` or
    :meth:`as_config_dict()`.
    """

    def as_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_private_fields: bool = False,
        include_class_name: bool = False,
        json_safe: bool = False,
        recurse: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert into a regular Python dictionary.

        :param exclude_none: Don't include values that are ``None``.
        :param exclude_private_fields: Don't include private fields.
        :param include_class_name: Include a field for the name of the class.
        :param json_safe: Output only JSON-safe types.
        :param recurse: Recurse into fields that are also configs/dataclasses.
        """

        def iter_fields(d) -> Generator[Tuple[str, Any], None, None]:
            for field in fields(d):
                value = getattr(d, field.name)
                if exclude_none and value is None:
                    continue
                elif exclude_private_fields and field.name.startswith("_"):
                    continue
                else:
                    yield (field.name, value)

        def as_dict(d: Any, recurse: bool = True) -> Any:
            if is_dataclass(d):
                if recurse:
                    out = {k: as_dict(v) for k, v in iter_fields(d)}
                else:
                    out = {k: v for k, v in iter_fields(d)}
                if include_class_name:
                    out[self.CLASS_NAME_FIELD] = d.__class__.__name__
                return out
            elif isinstance(d, dict):
                return {k: as_dict(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple, set)):
                if json_safe:
                    return [as_dict(x) for x in d]
                else:
                    return d.__class__((as_dict(x) for x in d))
            elif d is None or isinstance(d, (float, int, bool, str)):
                return d
            elif json_safe:
                return str(d)
            else:
                return d

        return as_dict(self, recurse=recurse)

    def as_config_dict(self) -> Dict[str, Any]:
        """
        A convenience wrapper around :meth:`as_dict()` for creating JSON-safe dictionaries suitable
        for recording the config.
        """
        return self.as_dict(
            exclude_none=True,
            exclude_private_fields=True,
            include_class_name=True,
            json_safe=True,
            recurse=True,
        )

    def apply(self, func: Callable[["Config"], None]):
        """
        Recursively apply a function to every config instance field, including ``self``.

        :param func: The function to apply.
        """

        def apply(d):
            if isinstance(d, Config):
                func(d)

            if is_dataclass(d):
                for field in fields(d):
                    value = getattr(d, field.name)
                    apply(value)
            elif isinstance(d, dict):
                for value in d.values():
                    apply(value)
            elif isinstance(d, (list, tuple, set)):
                for x in d:
                    apply(x)

        apply(self)

    def validate(self):
        """
        Validate fields in ``self``. This may modify ``self`` in-place.
        """
        pass

    def merge(self, dotlist: List[str], prefix: Optional[str] = None, strict: bool = True) -> Self:
        """
        Merge self with fields from a "dotlist", creating a new object.

        :param dotlist: A list of field attributes with dot notation, e.g. ``foo.bar=1``.
        :param prefix: Only use override items in the dotlist that start with a given prefix name,
            and strip that prefix (including the subsequent ".") before applying the overrides.
        :param strict: Parse the dotlist strictly.
        """
        try:
            dotlist = _clean_opts(dotlist)
            if prefix is not None:
                dotlist = [
                    o.replace(f"{prefix}.", "", 1) for o in dotlist if o.startswith(f"{prefix}.")
                ]
            if not strict:
                field_names = set(f.name for f in fields(self))
                dotlist = [
                    o
                    for o in dotlist
                    if any(
                        [
                            o.startswith(f"{name}=") or o.startswith(f"{name}.")
                            for name in field_names
                        ]
                    )
                ]
            merge_fields = om.from_dotlist(dotlist)
            merged = om.merge(self, merge_fields)
            out = cast(Self, om.to_object(merged))
            out.apply(lambda c: c.validate())
            return out
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))

    def replace(self, **changes) -> Self:
        """
        Creates a new object of the same type, replacing fields with values from ``changes``.
        """
        return replace(self, **changes)

    @classmethod
    def from_dict(cls: Type[C], data: Dict[str, Any], overrides: Optional[List[str]] = None) -> C:
        """
        Initialize from a regular Python dictionary.

        :param data: A Python dictionary.
        :param overrides: A list of field overrides with dot notation, e.g. ``foo.bar=1``.
        """
        try:
            schema = om.structured(cls)
            conf = om.merge(schema, data)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(_clean_opts(overrides)))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))


def _clean_opts(opts: List[str]) -> List[str]:
    return [_clean_opt(s) for s in opts]


def _clean_opt(arg: str) -> str:
    if "=" not in arg:
        arg = f"{arg}=True"
    name, val = arg.split("=", 1)
    name = name.strip("-").replace("-", "_")
    return f"{name}={val}"


class DType(StrEnum):
    """
    An enumeration of supported PyTorch data types.
    """

    float32 = "float32"
    bfloat16 = "bfloat16"

    @classmethod
    def from_pt(cls, dtype: torch.dtype) -> "DType":
        if dtype == torch.float32:
            return DType.float32
        elif dtype == torch.bfloat16:
            return DType.bfloat16
        else:
            raise NotImplementedError(dtype)

    def as_pt(self) -> torch.dtype:
        return getattr(torch, self)


class LayerNormType(StrEnum):
    default = "default"
    """
    The default LayerNorm implementation, equivalent to PyTorch's built-in version.
    """

    low_precision = "low_precision"
    """
    A low-precision version of the default LayerNorm.
    """

    rms = "rms"
    """
    An RMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"

    llama = "llama"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Llama.
    """


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


@dataclass
class ModelConfig(Config):
    """
    OLMo (model) configuration.
    """

    # Note that the defaults for these attributes are equivalent to the base GPT2 model.

    d_model: int = 768
    """
    The hidden size of the model.
    """

    n_heads: int = 12
    """
    The number of self-attention heads.
    """

    n_kv_heads: Optional[int] = None
    """
    The number of heads to use for keys and values. Defaults to `n_heads`.
    Set this to ``None`` or ``n_heads`` for normal multi-head attention.
    Set this to 1 for multi-query attention.
    Set it to some in-between value for Llama2-style grouped query attention.
    """

    clip_qkv: Optional[float] = None
    """
    Clip QKV to this value when set.
    """

    n_layers: int = 12
    """
    The number of layers/blocks.
    """

    mlp_ratio: int = 4
    """
    The ratio of the inner MLP dimensionality to ``d_model``.
    This is only used when ``mlp_hidden_size`` is not set.
    """

    mlp_hidden_size: Optional[int] = None
    """
    Set the exact hidden size for the MLP. Otherwise the inner MLP hidden size will be set to `mlp_ratio * d_model`.
    """

    activation_type: ActivationType = ActivationType.swiglu
    """
    The activation function to use within the MLP layers.
    """

    block_type: BlockType = BlockType.sequential
    """
    The transformer block implementation.
    """

    block_group_size: int = 1
    """
    The number of blocks to group together into a single parent block.
    This has no affect on the number of parameters in the model and is only used to wrap groups
    of blocks together with a single FSDP wrapper during training.
    """

    alibi: bool = False
    """
    If ``True``, use ALiBi embeddings. Mutually exclusive with ``rope``.
    """

    alibi_bias_max: float = 8.0
    """
    Maximum absolute value of ALiBi bias.
    """

    rope: bool = False
    """
    Use rotary positional embeddings (RoPE). Mutually exclusive with ``alibi``.
    """

    rope_full_precision: bool = True
    """
    If ``True``, apply RoPE embeddings at full precision regardless of the input type. Otherwise,
    apply RoPE at the precision of the input.
    """

    rope_theta: int = 10_000
    """
    The theta setting for RoPE.
    """

    flash_attention: bool = False
    """
    If ``True``, use ``FlashAttention``.
    """

    attention_dropout: float = 0.1
    """
    The dropout probability within the attention modules.
    """

    multi_query_attention: Optional[bool] = None
    """
    Deprecated. Use n_kv_heads instead.
    """

    attention_layer_norm: bool = False
    """
    Apply layer norm to the keys and queries within the attention mechanism.
    This can help stabilize training.
    """

    residual_dropout: float = 0.1
    """
    The dropout probability for the MLP and attention output within each block.
    """

    embedding_dropout: float = 0.1
    """
    The dropout probability for embeddings.
    """

    embedding_layer_norm: bool = False
    """
    Apply layer norm directly to the embeddings.
    """

    layer_norm_type: LayerNormType = LayerNormType.default
    """
    The layernorm implementation to use.
    """

    layer_norm_with_affine: bool = True
    """
    Whether to include bias and weight parameters for the layer norms.
    This only affects layer norms that are immediately followed by a linear layer in the forward pass,
    so everything except QK-norms. To turn off affines for QK norms as well, set :attr:`attention_layer_norm_with_affine`
    to ``False``.
    """

    layer_norm_eps: float = 1e-05

    attention_layer_norm_with_affine: bool = True
    """
    Toggle affine transform for the QK norms.
    """

    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """

    include_bias: bool = True
    """
    Whether or not to include bias parameters in linear layers.
    In PaLM, they got rid of all bias terms because they found that large
    models tend to have near 0 bias terms anyway.
    """

    bias_for_layer_norm: Optional[bool] = None
    """
    Whether or not to include bias parameters in layer norm.
    This is separate from the include_bias parameter, because of a ROCm crash when biases are disabled in
    layer norm.
    When this is None (the default), it inherits the setting from include_bias.
    """

    scale_logits: bool = False
    """
    If ``True``, scale the output logits by ``1 / sqrt(d_model)``.
    """

    vocab_size: int = 50257
    """
    Vocabulary size of the model.
    """

    embedding_size: Optional[int] = 50304
    """
    The number of embeddings, i.e. the number of tokens. If set to ``None`` it will default
    to ``vocab_size``. If ``vocab_size`` is not a multiple of 128, setting this to the
    next multiple of 128 that's greater than ``vocab_size`` can improve throughput
    substantially.
    """

    weight_tying: bool = True
    """
    Whether to tie output linear weights to the input embedding.
    """

    eos_token_id: int = 50256
    """
    The ID of the end-of-sentence special token.
    """

    pad_token_id: int = 50256
    """
    The ID of the token to use for padding. Defaults to the ID of the EOS token.
    """

    init_device: Optional[str] = None
    """
    The torch device to use when initializing the model parameters, e.g. "cpu", "cuda:0", "meta".
    """

    init_fn: InitFnType = InitFnType.normal
    """
    The weight initialization strategy.
    """

    init_std: float = 0.02
    """
    The standard deviation to use when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal".
    """

    init_cutoff_factor: Optional[float] = None
    """
    A positive factor used to scale the cutoff values when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal". Setting this to None means values are not cutoff.
    """

    precision: Optional[str] = None
    """
    Precision used to train/evaluate with. You shouldn't set this directly.
    See :data:`TrainConfig.precision` instead.
    """

    scale_emb_init: bool = False
    """
    If ``True``, embeddings are scaled up by ``sqrt(d_model)`` during initialization.
    Currently this is only used with `full_megatron` init when ``emb_init_std`` is unset.
    """

    emb_init_std: Optional[float] = None
    """
    Override the standard deviation to use when initializing the embedding weights.
    """

    norm_after: bool = False
    """
    Apply norm after the attention/feedforward layers rather than before, as introduced in the Swin transformer paper (Liu et al).
    """

    # muP parameters

    use_mup: bool = False
    """
    Whether the model is being parametrized in mup or standard parametrization.
    """

    # TODO: decide upon format; may require cached_path, etc.
    mup_base_shapes: Optional[str] = None
    """
    Optional path to base shapes in case of muP.
    # TODO: improve description
    """

    mup_query_zero_init: bool = False

    mup_base_n_heads: Optional[int] = None
    """
    The number of self-attention heads in the base muP model. This must be set when using muP.
    """

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            if self.multi_query_attention is True:
                return 1
            else:
                return self.n_heads
        else:
            if self.multi_query_attention is None:
                return self.n_kv_heads
            if self.multi_query_attention:
                n_kv_heads_should_be = 1
            else:
                n_kv_heads_should_be = self.n_heads
            if self.n_kv_heads == n_kv_heads_should_be:
                return n_kv_heads_should_be
            else:
                raise OLMoConfigurationError(
                    "You can't set `multi_query_attention` and `n_kv_heads` at the same time."
                )

