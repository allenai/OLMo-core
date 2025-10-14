import pytest
import torch
from olmo_core.optim import WSDS
from olmo_core.exceptions import OLMoConfigurationError


class TestWSDSScheduler:
    """Test suite for WSDS (Warmup-Stable-Decay-Simplified) scheduler."""

    def test_basic_single_period_with_warmup(self):
        """Test a single period with warmup, stable, and decay phases."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup=100,
            decay=200,
            warmup_min_lr=0.0,
            decay_min_lr=0.0,
        )
        initial_lr = 1.0
        
        # Test warmup phase (0-100)
        assert scheduler.get_lr(initial_lr, 0, 1000) == 0.0  # Start of warmup
        lr_mid_warmup = scheduler.get_lr(initial_lr, 50, 1000)
        assert 0.0 < lr_mid_warmup < initial_lr  # Increasing
        assert scheduler.get_lr(initial_lr, 100, 1000) == pytest.approx(initial_lr)  # End of warmup
        
        # Test stable phase (100-700)
        assert scheduler.get_lr(initial_lr, 200, 1000) == initial_lr
        assert scheduler.get_lr(initial_lr, 500, 1000) == initial_lr
        assert scheduler.get_lr(initial_lr, 799, 1000) == initial_lr
        
        # Test decay phase (800-1000)
        lr_mid_decay = scheduler.get_lr(initial_lr, 900, 1000)
        assert 0.0 < lr_mid_decay < initial_lr  # Decreasing
        assert scheduler.get_lr(initial_lr, 1000, 1000) == 0.0  # End of period

    def test_multiple_periods_with_reset(self):
        """Test that LR resets to peak at the start of each new period."""
        scheduler = WSDS(
            period_lengths=[1000, 1000],
            warmup=100,
            decay=200,
            warmup_min_lr=0.0,
            decay_min_lr=0.0,
        )
        initial_lr = 1.0
        
        # End of first period
        assert scheduler.get_lr(initial_lr, 1000, 2000) == 0.0
        
        # Start of second period - should reset to peak
        assert scheduler.get_lr(initial_lr, 1001, 2000) == initial_lr
        
        # Middle of second period stable phase
        assert scheduler.get_lr(initial_lr, 1400, 2000) == initial_lr
        
        # End of second period
        assert scheduler.get_lr(initial_lr, 2000, 2000) == 0.0

    def test_warmup_fraction(self):
        """Test warmup specified as a fraction."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup_fraction=0.1,  # 100 steps
            decay_fraction=0.2,   # 200 steps
            warmup_min_lr=0.0,
            decay_min_lr=0.0,
        )
        initial_lr = 1.0
        
        # Verify phases are correct length
        assert scheduler.get_lr(initial_lr, 0, 1000) == 0.0  # Start warmup
        assert scheduler.get_lr(initial_lr, 100, 1000) == pytest.approx(initial_lr)  # End warmup
        assert scheduler.get_lr(initial_lr, 500, 1000) == initial_lr  # Stable
        assert scheduler.get_lr(initial_lr, 1000, 1000) == 0.0  # End decay

    def test_decay_fraction(self):
        """Test decay specified as a fraction."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup=50,
            decay_fraction=0.3,  # 300 steps
            warmup_min_lr=0.0,
            decay_min_lr=0.0,
        )
        initial_lr = 1.0
        
        # Stable phase should be 1000 - 50 - 300 = 650 steps
        # Phases: warmup [0, 50), stable [50, 700), decay [700, 1000]
        assert scheduler.get_lr(initial_lr, 50, 1000) == initial_lr  # Start of stable
        assert scheduler.get_lr(initial_lr, 699, 1000) == initial_lr  # End stable
        assert scheduler.get_lr(initial_lr, 700, 1000) == initial_lr  # Still stable (boundary)
        lr_decay = scheduler.get_lr(initial_lr, 701, 1000)  # First step of decay
        assert 0.0 < lr_decay < initial_lr  # In decay
        assert scheduler.get_lr(initial_lr, 1000, 1000) == 0.0  # End of period

    def test_no_warmup_in_subsequent_periods(self):
        """Test that only period 0 has warmup."""
        scheduler = WSDS(
            period_lengths=[1000, 1000],
            warmup=100,
            decay=200,
        )
        initial_lr = 1.0
        
        # Period 1 starts immediately at peak LR (no warmup)
        assert scheduler.get_lr(initial_lr, 1001, 2000) == initial_lr

    def test_beyond_all_periods(self):
        """Test behavior when current step exceeds all periods."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup=100,
            decay=200,
            decay_min_lr=0.0,
        )
        initial_lr = 1.0
        
        # Beyond all periods, should stay at decay_min_lr
        assert scheduler.get_lr(initial_lr, 1001, 1000) == 0.0
        assert scheduler.get_lr(initial_lr, 5000, 1000) == 0.0

    def test_non_zero_decay_min_lr(self):
        """Test that decay_min_lr can be non-zero."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup=100,
            decay=200,
            decay_min_lr=0.1,
        )
        initial_lr = 1.0
        
        # End of decay should reach decay_min_lr
        assert scheduler.get_lr(initial_lr, 1000, 1000) == pytest.approx(0.1)
        
        # Beyond all periods should stay at decay_min_lr
        assert scheduler.get_lr(initial_lr, 1500, 1000) == 0.1

    def test_non_zero_warmup_min_lr(self):
        """Test that warmup can start from non-zero LR."""
        scheduler = WSDS(
            period_lengths=[1000],
            warmup=100,
            decay=200,
            warmup_min_lr=0.1,
        )
        initial_lr = 1.0
        
        # Start of warmup should be at warmup_min_lr
        assert scheduler.get_lr(initial_lr, 0, 1000) == 0.1

    def test_error_empty_period_lengths(self):
        """Test that empty period_lengths raises error."""
        with pytest.raises(OLMoConfigurationError, match="must be provided and non-empty"):
            WSDS(period_lengths=[])

    def test_error_negative_period_length(self):
        """Test that negative period lengths raise error."""
        with pytest.raises(OLMoConfigurationError, match="must be > 0"):
            WSDS(period_lengths=[1000, -500])

    def test_error_both_warmup_specified(self):
        """Test that specifying both warmup and warmup_fraction raises error."""
        with pytest.raises(OLMoConfigurationError, match="Exactly one"):
            WSDS(
                period_lengths=[1000],
                warmup=100,
                warmup_fraction=0.1,
                decay=200,
            )

    def test_error_neither_warmup_specified(self):
        """Test that specifying neither warmup nor warmup_fraction raises error."""
        with pytest.raises(OLMoConfigurationError, match="Exactly one"):
            WSDS(
                period_lengths=[1000],
                decay=200,
            )

    def test_error_both_decay_specified(self):
        """Test that specifying both decay and decay_fraction raises error."""
        with pytest.raises(OLMoConfigurationError, match="Exactly one"):
            WSDS(
                period_lengths=[1000],
                warmup=100,
                decay=200,
                decay_fraction=0.2,
            )

    def test_error_neither_decay_specified(self):
        """Test that specifying neither decay nor decay_fraction raises error."""
        with pytest.raises(OLMoConfigurationError, match="Exactly one"):
            WSDS(
                period_lengths=[1000],
                warmup=100,
            )

    def test_error_warmup_fraction_out_of_range(self):
        """Test that warmup_fraction must be in [0, 1]."""
        with pytest.raises(OLMoConfigurationError, match="must be in"):
            WSDS(
                period_lengths=[1000],
                warmup_fraction=1.5,
                decay=200,
            )

    def test_error_decay_fraction_out_of_range(self):
        """Test that decay_fraction must be in [0, 1]."""
        with pytest.raises(OLMoConfigurationError, match="must be in"):
            WSDS(
                period_lengths=[1000],
                warmup=100,
                decay_fraction=-0.1,
            )

    def test_error_warmup_plus_decay_exceeds_first_period(self):
        """Test that warmup + decay exceeding first period raises error."""
        with pytest.raises(OLMoConfigurationError, match="exceeds period length"):
            WSDS(
                period_lengths=[1000],
                warmup=600,
                decay=500,  # 600 + 500 = 1100 > 1000
            )

    def test_error_decay_exceeds_subsequent_period(self):
        """Test that decay exceeding any period length raises error."""
        with pytest.raises(OLMoConfigurationError, match="Period 1.*exceeds period length"):
            WSDS(
                period_lengths=[1000, 500],
                warmup=100,
                decay=600,  # 600 > 500 for period 1
            )

    @pytest.mark.parametrize("num_periods", [1, 2, 5])
    def test_variable_period_counts(self, num_periods):
        """Test scheduler works with different numbers of periods."""
        scheduler = WSDS(
            period_lengths=[1000] * num_periods,
            warmup=100,
            decay=200,
        )
        initial_lr = 1.0
        
        # Test each period ends at 0
        for i in range(num_periods):
            period_end = (i + 1) * 1000
            assert scheduler.get_lr(initial_lr, period_end, period_end) == 0.0

    @pytest.mark.parametrize("period_length", [100, 1000, 10000])
    def test_variable_period_lengths(self, period_length):
        """Test scheduler works with different period lengths."""
        warmup = int(0.1 * period_length)
        decay = int(0.2 * period_length)
        
        scheduler = WSDS(
            period_lengths=[period_length],
            warmup=warmup,
            decay=decay,
        )
        initial_lr = 1.0
        
        # Basic sanity checks
        assert scheduler.get_lr(initial_lr, 0, period_length) == 0.0
        assert scheduler.get_lr(initial_lr, period_length // 2, period_length) == initial_lr
        assert scheduler.get_lr(initial_lr, period_length, period_length) == 0.0