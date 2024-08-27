from olmo_core.data import DataMix, TokenizerName
from olmo_core.io import file_exists


def test_olmoe_mix():
    mix = DataMix.OLMoE_mix_0824.build("s3://ai2-llm", TokenizerName.dolma2)
    assert (
        mix[-1]
        == "s3://ai2-llm/preprocessed/olmo-mix/danyh-compiled-v1_7/documents/wiki/allenai/dolma2-tokenizer/part-1-00000.npy"
    )
    assert file_exists(mix[-1])


def test_dolma17_mix():
    mix = DataMix.dolma17.build("s3://ai2-llm", TokenizerName.gpt_neox_olmo_dolma_v1_5)
    assert (
        mix[-1]
        == "s3://ai2-llm/preprocessed/olmo-mix/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001/cc_en_tail/gpt-neox-olmo-dolma-v1_5/part-092-00000.npy"
    )
    assert file_exists(mix[-1])
