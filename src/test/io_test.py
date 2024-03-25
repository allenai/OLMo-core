from olmo_core.io import deserialize_from_tensor, serialize_to_tensor


def test_serde_from_tensor():
    data = {"a": (1, 2)}
    assert deserialize_from_tensor(serialize_to_tensor(data)) == data
