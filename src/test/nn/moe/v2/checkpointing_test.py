from olmo_core.nn.moe.v2.checkpointing import (
    checkpoint_forward_context,
    checkpoint_recompute_context,
    get_rowwise_checkpoint_state,
    is_activation_checkpointing,
    is_checkpoint_forwarding,
    is_checkpoint_recomputing,
)


def test_forward_context_tracks_depth():
    assert not is_checkpoint_forwarding()
    with checkpoint_forward_context():
        assert is_checkpoint_forwarding()
        with checkpoint_forward_context():  # nested
            assert is_checkpoint_forwarding()
        assert is_checkpoint_forwarding()  # still inside the outer context
    assert not is_checkpoint_forwarding()  # restored


def test_recompute_context_tracks_depth():
    with checkpoint_recompute_context():
        assert is_checkpoint_recomputing()
        with checkpoint_recompute_context():  # nested
            assert is_checkpoint_recomputing()
        assert is_checkpoint_recomputing()
    assert not is_checkpoint_recomputing()


def test_is_activation_checkpointing():
    assert not is_activation_checkpointing()
    with checkpoint_forward_context():
        assert is_activation_checkpointing()
    with checkpoint_recompute_context():
        assert is_activation_checkpointing()


def test_get_rowwise_checkpoint_state():
    # On the initial checkpointed forward: active, and outputs should be saved.
    with checkpoint_forward_context():
        assert get_rowwise_checkpoint_state() == (True, True)
    # During recompute: active, but outputs should not be re-saved.
    with checkpoint_recompute_context():
        assert get_rowwise_checkpoint_state() == (True, False)
