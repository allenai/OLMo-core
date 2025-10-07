"""
Overview
--------

A composable data loading API for fixed sequence length text data.

::

    ┌─────────────┐       ┌────────────────┐       ┌──────────────────────┐
    │ TokenSource │ ⇢ ⋯ ⇢ │ InstanceSource │ ⇢ ⋯ ⇢ │ ComposableDataLoader │
    └─────────────┘       └────────────────┘       └──────────────────────┘

This API consists of a series of simple, composable, elements, including:

1. :class:`TokenSource` and :class:`DocumentSource`: Token sources provide access to tokenized text data, while
   document sources are special token sources that also provide information on where the document boundaries are.
   Examples include:

   * :class:`InMemoryTokenSource` and :class:`InMemoryDocumentSource`: A simple token/document source that holds all tokens in memory.
   * :class:`ConcatenatedTokenSource` and :class:`ConcatenatedDocumentSource`: A token/document that combines multiple sources into one.
   * :class:`NumpyDocumentSource`: A document that reads tokens from one or more numpy source files, like those created
     from the dolma toolkit.
   * :class:`SamplingTokenSource` and :class:`SamplingDocumentSource`: A token/document source that samples tokens/documents
     from one or more other token/document sources.

2. :class:`InstanceSource`: Instance sources convert token sources (or in some case other instance sources)
   into fixed-length instances.
   Examples include:

   * :class:`ConcatAndChunkInstanceSource`: The simplest instance source that chunks up token sources
     without regard for document boundaries, just like the :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset`.
   * :class:`PackingInstanceSource`: An instance source that packs documents from one or more document
     sources into instances using an optimized packing algorithm.

3. :class:`ComposableDataLoader`: A data loader for OLMo-core's :class:`~olmo_core.train.Trainer` that takes
   one or more instance sources.

Reference
---------
"""

from .concat_and_chunk_instance_source import (
    ConcatAndChunkInstanceSource,
    ConcatAndChunkInstanceSourceConfig,
)
from .data_loader import (
    ComposableDataLoader,
    ComposableDataLoaderConfig,
    InstanceFilterConfig,
    ShuffleStrategy,
)
from .instance_source import Instance, InstanceSource, InstanceSourceConfig
from .numpy_document_source import (
    NumpyDocumentSource,
    NumpyDocumentSourceConfig,
    NumpyDocumentSourceMixConfig,
)
from .packing_instance_source import (
    LongDocStrategy,
    PackingInstanceSource,
    PackingInstanceSourceConfig,
)
from .sampling_document_source import (
    SamplingDocumentSource,
    SamplingDocumentSourceConfig,
)
from .sampling_token_source import SamplingTokenSource, SamplingTokenSourceConfig
from .token_source import (
    ConcatenatedDocumentSource,
    ConcatenatedTokenSource,
    DocumentSource,
    DocumentSourceConfig,
    InMemoryDocumentSource,
    InMemoryTokenSource,
    TokenRange,
    TokenSource,
    TokenSourceConfig,
)

__all__ = [
    # Base classes.
    "TokenSource",
    "TokenSourceConfig",
    "DocumentSource",
    "DocumentSourceConfig",
    "TokenRange",
    "InstanceSource",
    "InstanceSourceConfig",
    "Instance",
    "ComposableDataLoader",
    "ComposableDataLoaderConfig",
    # Token/document source implementations.
    "InMemoryTokenSource",
    "ConcatenatedTokenSource",
    "SamplingTokenSource",
    "SamplingTokenSourceConfig",
    "InMemoryDocumentSource",
    "ConcatenatedDocumentSource",
    "SamplingDocumentSource",
    "SamplingDocumentSourceConfig",
    "NumpyDocumentSource",
    "NumpyDocumentSourceConfig",
    "NumpyDocumentSourceMixConfig",
    # Instance source implementations.
    "ConcatAndChunkInstanceSource",
    "ConcatAndChunkInstanceSourceConfig",
    "PackingInstanceSource",
    "PackingInstanceSourceConfig",
    # Other types.
    "InstanceFilterConfig",
    "LongDocStrategy",
    "ShuffleStrategy",
]
