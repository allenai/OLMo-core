.. _data_mixing:

Data Mixing
===========

.. note::
   This guide discussions functionality that is specific to text-based data.

``olmo_core`` provides three ways to mix sets of pre-tokenized numpy files for training:

- Fine-grained, ratio-based sampling driven by :mod:`olmo_core.data.source_mixture` or
  the various composable mixing classes in :mod:`olmo_core.data.composable`.
- Predefined, catalog-style mixes exposed through :mod:`olmo_core.data.mixes`.
- Simple path/glob lists passed directly to a
  :class:`~olmo_core.data.numpy_dataset.NumpyDatasetConfig` or
  :class:`~olmo_core.data.composable.NumpyDocumentSource` if you prefer the
  :mod:`olmo_core.data.composable` API.

Fine-grained source mixtures
----------------------------

Use :class:`~olmo_core.data.source_mixture.SourceMixtureConfig` with
:class:`~olmo_core.data.source_mixture.SourceMixtureDatasetConfig`,
or the mixing classes in :mod:`olmo_core.data.composable`,
when you need precise control over per-source proportions, repetition limits, or source fractions.

Via the composable API
^^^^^^^^^^^^^^^^^^^^^^

The composable data loading API allows you to build fine-grained mixtures by mixing/sampling at various
levels: either the token-level (:class:`~olmo_core.data.composable.MixingTokenSource`),
document-level (:class:`~olmo_core.data.composable.MixingDocumentSource`),
or instance-level (:class:`~olmo_core.data.composable.MixingInstanceSource`).

Each of those classes has a similar API, which accepts a sequence of "source specs" with the following parameters:

- ``source`` – The underlying source to sample from.
  Either a :class:`~olmo_core.data.composable.TokenSource`, :class:`~olmo_core.data.composable.DocumentSource`,
  or :class:`~olmo_core.data.composable.InstanceSource` depending on the mixing class.
- ``ratio`` – The relative target ratio for the source in the final mixture.
- ``max_repetition_factor`` – The maximum amount of repetition allowed,
  expressed as a factor greater than or equal to 1.0.
  A factor of 1.0 means no repetition is allowed. A factor of 2.0 means each instance could be
  repeated at most once (i.e., seen twice).
- ``label`` – An optional label to assign the source for logging and debugging purposes.

In this example we'll demonstrate instance-level mixing with the "concat and chunk"
(:class:`~olmo_core.data.composable.ConcatAndChunkInstanceSource`) strategy for building instances.

::

    import functools as ft
    
    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import *
    
    tokenizer = TokenizerConfig.dolma2()
    sequence_length = 2048
    
    npy_instance_source = ft.partial(
        ConcatAndChunkInstanceSource.Config.from_npy,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )
    
    mix_config = MixingInstanceSource.Config(
        num_tokens=1_000_000_000,
        source_specs=[
            MixingInstanceSource.Spec.Config(
                source=npy_instance_source("/corpus/trex-facts/part-*.npy"),
                ratio=0.6,
                label="trex-facts",
            ),
            MixingInstanceSource.Spec.Config(
                source=npy_instance_source("/corpus/triceratops-facts/shard-*.npy"),
                ratio=0.3,
                label="triceratops-knowledge",
            ),
            MixingInstanceSource.Spec.Config(
                source=npy_instance_source("/corpus/stegosaurus-high-quality/*.npy"),
                ratio=0.1,
                label="stegosaurus-high-quality",
            ),
        ]
    )

Once you have your mix config, you can call :meth:`~olmo_core.data.composable.InstanceSourceConfig.build()`
on it to get a :class:`~olmo_core.data.composable.InstanceSource` that you can pass to a
:class:`~olmo_core.data.composable.ComposableDataLoader` or wrap in another instance source such as
the :class:`~olmo_core.data.composable.SamplingInstanceSource` to adjust the number of instances per epoch.

.. tip::
   See the :mod:`olmo_core.data.composable` module documentation for a more in-depth overview of this API.

Via the :mod:`~olmo_core.data.source_mixture` API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defining the sources
~~~~~~~~~~~~~~~~~~~~

Create one :class:`~olmo_core.data.source_mixture.SourceMixtureConfig` per data
source. Each config specifies where the token files live and how aggressively that
source should be sampled:

- ``target_ratio`` – Global proportion of tokens that should come from this source.
  All ratios across sources must sum to ``1.0``.
- ``paths`` – List of numpy ``.npy`` files that contain pre-tokenized data.
- ``max_repetition_ratio`` – Optional upsampling factor. Values above ``1.0`` allow
  the same file set to be repeated to hit the requested ratio when there are not enough
  unique tokens available.
- ``max_source_fraction`` – Caps how much of the underlying source population can be
  consumed, useful when you want to leave hold-out data untouched.

Creating the dataset plan
~~~~~~~~~~~~~~~~~~~~~~~~~

Wrap the per-source configs in a :class:`~olmo_core.data.source_mixture.SourceMixtureDatasetConfig`.
This object performs token counting, enforces the requested ratios, and produces a
:class:`~olmo_core.data.source_mixture.SourceMixtureDataset` that enumerates every
path to read during training.

Key parameters:

- ``requested_tokens`` – Minimum number of tokens you want the mixture to deliver.
  The builder rounds up so that the dataset contains an integer number of training
  instances at your target sequence length.
- ``global_batch_size`` – Total tokens consumed per optimizer step across all ranks.
  This must be a multiple of the model ``sequence_length``.
- ``processes`` – Number of worker threads used to scan file sizes in parallel.
- ``seed`` – Controls deterministic ordering when fractional tokens are rounded.
- ``render_tables`` / ``quiet`` – Toggle rich tables that summarize the final mix.

Typical usage together with the fixed-sequence dataset config::

    from olmo_core.data import NumpyFSLDatasetConfig
    from olmo_core.data.source_mixture import (
        SourceMixtureConfig,
        SourceMixtureDatasetConfig,
        SourceMixtureList,
    )
    from olmo_core.train import TokenizerConfig

    sequence_length = 2048
    global_batch_size = sequence_length * 512  # tokens per optimizer step

    mix_config = SourceMixtureDatasetConfig(
        source_list=SourceMixtureList([
            SourceMixtureConfig(
                source_name="trex-facts",
                target_ratio=0.6,
                paths=["/corpus/trex-facts/part-*.npy"],
            ),
            SourceMixtureConfig(
                source_name="triceratops-knowledge",
                target_ratio=0.3,
                paths=["/corpus/triceratops-facts/shard-*.npy"],
                max_repetition_ratio=1.5,
            ),
            SourceMixtureConfig(
                source_name="stegosaurus-high-quality",
                target_ratio=0.1,
                paths=["/corpus/stegosaurus-high-quality/*.npy"],
                max_source_fraction=0.25,
            ),
        ]),
        requested_tokens=1_000_000_000,
        global_batch_size=global_batch_size,
        processes=16,
        quiet=False,
    )

    dataset_cfg = NumpyFSLDatasetConfig.from_src_mix(
        mix_config,
        tokenizer=TokenizerConfig.dolma2(),
        sequence_length=sequence_length,
    )

When the trainer calls ``dataset_cfg.build()``, the mix configuration counts tokens
for each path, applies the Hamilton apportionment scheme to keep ratios precise,
logs summary tables (unless ``quiet=True``), and supplies file weights to
:class:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetMixture`.

If the requested ratios cannot be met because a source is too small and repetition
is disabled, ``build()`` raises :class:`~olmo_core.exceptions.OLMoConfigurationError`
so you can adjust either the ratios or ``max_repetition_ratio``.

Source mixture datasets are currently ony compatible with :class:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetMixture`
(no padding, packing, or VSL support).

Predefined data mixes
---------------------

For simpler scenarios, rely on the curated mixes defined in
:mod:`olmo_core.data.mixes`. The :class:`~olmo_core.data.mixes.DataMix` enumeration
encodes a set of text manifests (``*.txt``) that list shard labels and relative
paths.

Example::

    from olmo_core.data import DataMix, NumpyFSLDatasetConfig
    from olmo_core.train import TokenizerConfig

    dataset_cfg = NumpyFSLDatasetConfig(
        mix=DataMix.OLMoE_mix_0824,
        mix_base_dir="s3://ai2-llm",
        tokenizer=TokenizerConfig(identifier="dolma2-tokenizer"),
        sequence_length=2048,
    )

    dataset = dataset_cfg.build()

When ``mix`` is set, ``NumpyFSLDatasetConfig`` loads the manifest, patches the
``{TOKENIZER}`` placeholder to match the active tokenizer identifier (with special
cases handled for :class:`~olmo_core.data.tokenizer.TokenizerName` variants), and
injects any shard labels into the dataset metadata. You only need to supply the base
location (for example an S3 prefix or shared filesystem path).

Extending the catalog
^^^^^^^^^^^^^^^^^^^^^

If you need to register a new preset mix, subclass :class:`~olmo_core.data.mixes.DataMixBase`
inside your project, add an enum value, and provide a matching ``.txt`` manifest
with ``label,path`` rows stored under ``olmo_core/data/mixes``. The rest of the
pipeline can treat your custom enum exactly like the built-in :class:`DataMix`
options.

Simple path/glob lists
----------------------

The most direct option is to point :class:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig`
at an explicit list of token files. This is ideal when your data already lives in a
single directory or you simply want all files treated uniformly without additional
weighting.

Pass absolute or relative paths via the ``paths`` field. If you prefer glob
patterns, either set ``expand_glob=True`` or use the convenience
:meth:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig.glob` constructor, which
defers pattern expansion until ``build()`` so validation can happen during startup.

Example::

    from olmo_core.data import NumpyFSLDatasetConfig
    from olmo_core.train import TokenizerConfig

    dataset_cfg = NumpyFSLDatasetConfig(
        paths=[
            "/datasets/run1/shard00.npy",
            "/datasets/run1/shard01.npy",
        ],
        tokenizer=TokenizerConfig.dolma2(),
        sequence_length=2048,
    )

    # Equivalent glob-based declaration
    dataset_cfg = NumpyFSLDatasetConfig.glob(
        "/datasets/run1/shard*.npy",
        tokenizer=TokenizerConfig.dolma2(),
        sequence_length=2048,
    )

Additional knobs:

- ``label_mask_paths`` can point to matching numpy boolean files that provide per-token label
  masks (to exclude certain tokens from loss computation during training).
- ``metadata`` accepts per-path dictionaries that are returned with each instance
  when ``include_instance_metadata=True`` (the default).


Choosing an approach
--------------------

Use the fine-grained source mixture when you need deterministic token budgets,
custom repetition logic, or transparent ratio auditing. Reach for the predefined
mixes when an existing manifest already captures the blend you want or when you
prefer to manage mixes declaratively without writing code. Use simple path lists
for one-off experiments or small datasets.
