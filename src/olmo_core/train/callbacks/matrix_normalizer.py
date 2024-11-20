from dataclasses import dataclass

from .callback import Callback


@dataclass
class MatrixNormalizerCallback(Callback):
    """
    A callback to be used in conjunction with :class:`~olmo_core.nn.transformer.NormalizedTransformer`
    (nGPT) models to re-normalize the weight matrices after each optimizer step.
    """

    def post_train_batch(self):
        self.trainer.model.normalize_matrices()
