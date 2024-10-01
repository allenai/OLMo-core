from .custom_data_loader import CustomDataLoader


def test_custom_data_loader(tmp_path):
    data_loader = CustomDataLoader(
        sequence_length=128,
        vocab_size=1024,
        work_dir=tmp_path,
        global_batch_size=512,
        total_batches=100,
    )
    data_loader.reshuffle()

    batches_processed = 0
    for batch in data_loader:
        batches_processed += 1
        assert batch["input_ids"].numel() == 512
        if batches_processed > 10:
            break

    state_dict = data_loader.state_dict()
    data_loader.reset()
    data_loader.load_state_dict(state_dict)
    assert data_loader.batches_processed == batches_processed

    batches_processed = 0
    for batch in data_loader:
        batches_processed += 1
        assert batch["input_ids"].numel() == 512

    assert batches_processed == 100
