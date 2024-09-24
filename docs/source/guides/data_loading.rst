Data loading
============

Data preparation
----------------

OLMo-core's builtin data loading functionality requires you to pre-tokenize your data into 1D numpy arrays of token IDs. These arrays should include all special tokens already -- such as "end of sentence" (EOS) tokens -- except for padding tokens.

For example::

    import numpy as np

    documents = ["Hello, World!", "The quick brown fox jumped over the fence"]

    # Tokenize documents
    token_ids = []
    for doc in documents:
        token_ids.extend(tokenizer.encode(doc))

    # Write token IDs to disk
    data_mmap = np.memmap("data001.npy", mode="w+", dtype=np.uint32, shape=(len(token_ids),))
    data_mmap[:] = token_ids
    data_mmap.flush()

.. seealso::
    The `dolma <https://github.com/allenai/dolma>`_ project includes an optimized toolkit for pre-processing data into this format.

Train data loading
------------------

Once you're data is pre-processed as above there are several different strategies available for loading that data for training.

Concat and chunk
~~~~~~~~~~~~~~~~

The simplest strategy is the "concat and chunk" approach, which means all tokenized documents are concatenated together and then chunked into training instances of the desired sequence length.
To use this method you just need to pass a :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset`
dataset to your :class:`~olmo_core.train.Trainer` with the paths to your tokenized numpy data files.

While this strategy is simple and efficient, it does have several downsides.
For one, documents may end up fragmented across multiple training instances.
Second, since each training instance may be composed of multiple documents, the model will be attending to tokens across different documents, which could potentially have adverse affects on the model (see `Zhao et al. (2024) <Zhao et al 2024_>`_ for example).

Concat and chunk with intra-document masking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The latter point downside, however, can be addressed by using `intra-document masking <intra-document masking_>`_. This requires you set ``generate_doc_lengths=True`` in your :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset` and use a model implementation that accepts the parameters ``doc_lens`` and ``max_doc_lens`` to its ``forward()`` method. See the :class:`~olmo_core.nn.transformer.Transformer` model implementation for example.

Variable sequence length training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A promising alternative to the simple fixed sequence length "concat and chunk" strategy is variable sequence length training through "dataset decomposition" (`Pouransari et al. (2024) <Pouransari et al 2024_>`_).
With dataset decomposition, every training instance is a unique subset of tokens from a single document. Therefore there's no need for intra-document masking.
You can use this approach by passing a :class:`~olmo_core.data.numpy_dataset.NumpyVSLDataset` to your training with the paths to your numpy data files.

This requires you set a ``min_sequence_length`` and ``max_sequence_length`` which must both be powers of 2 (e.g. 256 and 4096). Each training batch will be composed of instances of the same sequence length such that the total number of tokens in the batch is equal to your :data:`~olmo_core.train.Trainer.global_batch_size`.

You can also configure a :class:`~olmo_core.data.numpy_dataset.VSLCurriculum` to control the sampling probability of different sequence lengths over the course of an epoch.


.. _intra-document masking: https://www.semanticscholar.org/paper/c4673ed74c1f25a67c346dfddb4944e64b0d00a6
.. _Zhao et al 2024: `intra-document masking`_

.. _dataset decomposition: https://www.semanticscholar.org/paper/a14648abc56c602396634609e911f0ec071e43c1
.. _Pouransari et al 2024: `dataset decomposition`_
