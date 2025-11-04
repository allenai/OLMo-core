import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open

from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.train.callbacks.gradient_dumper import GradientDumperCallback
from olmo_core.train.train_module import BasicTrainModule


class SimpleModel(nn.Module):
    """Simple model for testing gradient dumping."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(100, dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(self.embedding(x))


def create_mock_trainer(work_dir: Path, save_folder: str, global_step: int = 0):
    """Create a mock trainer for testing."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters())
    train_module = BasicTrainModule(model, optimizer, 128)

    trainer = MagicMock()
    trainer.work_dir = work_dir
    trainer.save_folder = save_folder
    trainer.save_overwrite = False
    trainer.global_step = global_step
    trainer.train_module = train_module

    return trainer


def test_gradient_dumper_disabled(tmp_path):
    """Test that disabled callback doesn't save anything."""
    trainer = create_mock_trainer(tmp_path, str(tmp_path / "save"))
    callback = GradientDumperCallback(enabled=False)
    callback.trainer = trainer

    # Set gradients
    for p in trainer.train_module.model.parameters():
        p.grad = torch.randn_like(p)

    callback.pre_optim_step()

    # Should not create any directories
    gradient_dir = trainer.work_dir / "gradient_dumper"
    assert not gradient_dir.exists()


def test_gradient_dumper_saves_all_gradients(tmp_path):
    """Test that callback saves all gradients when save_first_n is None."""
    trainer = create_mock_trainer(tmp_path, str(tmp_path / "save"))
    callback = GradientDumperCallback(enabled=True, start_step=0, step_interval=1)
    callback.trainer = trainer

    # Set gradients
    for p in trainer.train_module.model.parameters():
        p.grad = torch.randn_like(p)

    callback.pre_optim_step()

    # Check that directory was created
    step_dir = trainer.work_dir / "gradient_dumper" / "step0"
    assert step_dir.exists()

    # Check that files were saved
    files = list(step_dir.glob("*.safetensors"))
    assert len(files) > 0

    # Check that each saved file can be loaded
    for filepath in files:
        assert filepath.exists()
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            assert "gradient" in f.keys()
            grad = f.get_tensor("gradient")
            assert grad is not None


def test_gradient_dumper_saves_first_n(tmp_path):
    """Test that callback saves first N elements when save_first_n is set."""
    trainer = create_mock_trainer(tmp_path, str(tmp_path / "save"))
    callback = GradientDumperCallback(enabled=True, start_step=0, save_first_n=5)
    callback.trainer = trainer

    # Set gradients with specific shape
    for p in trainer.train_module.model.parameters():
        p.grad = torch.randn_like(p)

    callback.pre_optim_step()

    # Check that directory was created
    step_dir = trainer.work_dir / "gradient_dumper" / "step0"
    assert step_dir.exists()

    # Check that files were saved with _firstN suffix
    files = list(step_dir.glob("*_first*.safetensors"))
    assert len(files) > 0

    # Check that saved gradients have correct shape
    for filepath in files:
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            grad = f.get_tensor("gradient")
            assert grad is not None
            # Should be sliced to first N elements along first dimension
            assert grad.shape[0] <= 5


def test_gradient_dumper_step_filtering(tmp_path):
    """Test that callback respects start_step, end_step, and step_interval."""
    trainer = create_mock_trainer(tmp_path, str(tmp_path / "save"))
    callback = GradientDumperCallback(
        enabled=True, start_step=2, end_step=5, step_interval=2
    )
    callback.trainer = trainer

    # Set gradients
    for p in trainer.train_module.model.parameters():
        p.grad = torch.randn_like(p)

    step_dir = trainer.work_dir / "gradient_dumper"

    # Step 0: before start_step, should not save
    trainer.global_step = 0
    callback.pre_optim_step()
    assert not (step_dir / "step0").exists()

    # Step 2: at start_step, should save
    trainer.global_step = 2
    callback.pre_optim_step()
    assert (step_dir / "step2").exists()

    # Step 3: not at interval, should not save
    trainer.global_step = 3
    callback.pre_optim_step()
    assert not (step_dir / "step3").exists()

    # Step 4: at interval, should save
    trainer.global_step = 4
    callback.pre_optim_step()
    assert (step_dir / "step4").exists()

    # Step 6: after end_step, should not save
    trainer.global_step = 6
    callback.pre_optim_step()
    assert not (step_dir / "step6").exists()


def test_gradient_dumper_validation_error(tmp_path):
    """Test that callback raises error for invalid save_first_n."""
    trainer = create_mock_trainer(tmp_path, str(tmp_path / "save"))
    callback = GradientDumperCallback(enabled=True, save_first_n=0)
    callback.trainer = trainer

    # Set gradients
    for p in trainer.train_module.model.parameters():
        p.grad = torch.randn_like(p)

    with pytest.raises(ValueError, match="save_first_n must be positive"):
        callback.pre_optim_step()


def test_gradient_dumper_no_gradients(tmp_path):
    """Test that callback handles parameters with no gradients gracefully."""
    trainer = create_mock_trainer(tmp_path, str(tmp_path / "save"))
    callback = GradientDumperCallback(enabled=True, start_step=0)
    callback.trainer = trainer

    # Don't set gradients - parameters should have None gradients
    callback.pre_optim_step()

    # Should create directory but no files if no gradients
    step_dir = trainer.work_dir / "gradient_dumper" / "step0"
    # Directory should be created and empty
    if step_dir.exists():
        files = list(step_dir.glob("*.safetensors"))
        assert len(files) == 0


def test_gradient_dumper_upload_when_different(tmp_path):
    """Test that callback uploads to save_folder when different from work_dir."""
    work_dir = tmp_path / "work"
    save_folder = str(tmp_path / "save")
    trainer = create_mock_trainer(work_dir, save_folder)
    callback = GradientDumperCallback(enabled=True, start_step=0)
    callback.trainer = trainer

    # Set gradients
    for p in trainer.train_module.model.parameters():
        p.grad = torch.randn_like(p)

    with patch("olmo_core.train.callbacks.gradient_dumper.copy_dir") as mock_copy:
        callback.pre_optim_step()

        # Should create local directory
        step_dir = work_dir / "gradient_dumper" / "step0"
        assert step_dir.exists()

        # Should call copy_dir to upload
        assert mock_copy.called
        call_args = mock_copy.call_args
        assert str(call_args[0][0]) == str(step_dir)
        assert "gradient_dumper" in str(call_args[0][1])


def test_gradient_dumper_no_upload_when_same(tmp_path):
    """Test that callback doesn't upload when work_dir == save_folder."""
    save_folder = str(tmp_path)
    trainer = create_mock_trainer(tmp_path, save_folder)
    callback = GradientDumperCallback(enabled=True, start_step=0)
    callback.trainer = trainer

    # Set gradients
    for p in trainer.train_module.model.parameters():
        p.grad = torch.randn_like(p)

    with patch("olmo_core.train.callbacks.gradient_dumper.copy_dir") as mock_copy:
        callback.pre_optim_step()

        # Should create directory
        step_dir = tmp_path / "gradient_dumper" / "step0"
        assert step_dir.exists()

        # Should not call copy_dir since work_dir == save_folder
        assert not mock_copy.called


def run_gradient_dumper_distributed(work_dir, save_folder, save_first_n):
    """Helper function for distributed gradient dumper test."""
    from olmo_core.distributed.utils import get_rank

    trainer = create_mock_trainer(Path(work_dir), save_folder)
    callback = GradientDumperCallback(enabled=True, start_step=0, save_first_n=save_first_n)
    callback.trainer = trainer

    # Set gradients
    for p in trainer.train_module.model.parameters():
        p.grad = torch.randn_like(p)

    callback.pre_optim_step()

    # Check that files were saved
    step_dir = Path(work_dir) / "gradient_dumper" / "step0"
    assert step_dir.exists()

    if save_first_n is None:
        # Should save per-rank files
        rank = get_rank()
        files = list(step_dir.glob(f"rank{rank}_*.safetensors"))
        assert len(files) > 0
    else:
        # Only rank 0 should save files
        if get_rank() == 0:
            files = list(step_dir.glob("*_first*.safetensors"))
            assert len(files) > 0


@pytest.mark.parametrize("save_first_n", [None, 10])
@requires_multi_gpu
def test_gradient_dumper_distributed(tmp_path, save_first_n):
    """Test gradient dumper in distributed setting."""
    os.environ["OLMO_SHARED_FS"] = "1"

    run_distributed_test(
        run_gradient_dumper_distributed,
        func_args=(str(tmp_path / "work"), str(tmp_path / "save"), save_first_n),
        start_method="spawn",
    )


def test_gradient_dumper_caps_save_first_n(tmp_path):
    """Test that callback caps save_first_n when it exceeds dimension size."""
    trainer = create_mock_trainer(tmp_path, str(tmp_path / "save"))
    callback = GradientDumperCallback(enabled=True, start_step=0, save_first_n=1000)
    callback.trainer = trainer

    # Set gradients with actual parameter shapes
    # SimpleModel has parameters with shapes like (100, 8) and (8, 8)
    # save_first_n=1000 exceeds the first dimension size, so it should cap
    for p in trainer.train_module.model.parameters():
        p.grad = torch.randn_like(p)

    callback.pre_optim_step()

    # Should save successfully with capped size
    step_dir = trainer.work_dir / "gradient_dumper" / "step0"
    assert step_dir.exists()

    files = list(step_dir.glob("*_first*.safetensors"))
    assert len(files) > 0

    # Check that saved gradients are capped to actual dimension sizes
    for filepath in files:
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            grad = f.get_tensor("gradient")
            # Should be capped - first dimension should be <= actual parameter size
            # SimpleModel has max first dim of 100 (embedding), so should be <= 100
            assert grad.shape[0] <= 100

