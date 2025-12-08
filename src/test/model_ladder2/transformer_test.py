import pytest

from olmo_core.model_ladder2.transformer import TransformerLadderRun, TransformerSize


@pytest.mark.parametrize("size", TransformerSize)
def test_run_sizes(size: TransformerSize):
    run = TransformerLadderRun(
        size=size, chinchilla_multiple=1.0, sequence_length=8192, batch_size=8192 * 512
    )
    run.configure_model(50304)
