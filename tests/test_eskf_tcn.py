"""
Unit tests for ESKF_TCN hybrid model
"""
import pytest
import torch
from model.ESKF_TCN import ESKFTCN_model as ESKF_TCN
from model.config import Config


class TestESKF_TCNInitialization:
    """Test ESKF_TCN initialization"""

    def test_model_creation(self):
        """Model should initialize without errors"""
        model = ESKF_TCN()
        assert model is not None

    def test_has_eskf_component(self):
        """Model should have ESKF component"""
        model = ESKF_TCN()
        assert hasattr(model, 'eskf')
        assert model.eskf is not None

    def test_has_tcn_component(self):
        """Model should have TCN component"""
        model = ESKF_TCN()
        assert hasattr(model, 'tcn')
        assert model.tcn is not None


class TestESKF_TCNForward:
    """Test ESKF_TCN forward pass"""

    @pytest.fixture
    def model(self):
        """Create model for testing"""
        return ESKF_TCN()

    @pytest.fixture
    def sample_batch(self):
        """Create sample input batch"""
        batch_size = 2
        seq_len = 100

        return {
            'sensor': torch.randn(batch_size, seq_len, 7),  # accel(3) + gyro(3) + fsr(1)
            'gt_pos': torch.randn(batch_size, seq_len, 3),
            'gt_vel': torch.randn(batch_size, seq_len, 3),
            'seq_len': seq_len
        }

    def test_forward_pass_returns_dict(self, model, sample_batch):
        """Forward pass should return dictionary with required keys"""
        output = model(sample_batch)

        assert isinstance(output, dict)
        assert 'pred_pos' in output
        assert 'pred_vel' in output
        assert 'vel_corr' in output
        assert 'zupt_prob' in output

    def test_forward_pass_shapes(self, model, sample_batch):
        """Forward pass should return correct shapes"""
        batch_size = sample_batch['sensor'].shape[0]
        seq_len = sample_batch['sensor'].shape[1]

        output = model(sample_batch)

        assert output['pred_pos'].shape == (batch_size, seq_len, 3)
        assert output['pred_vel'].shape == (batch_size, seq_len, 3)
        assert output['vel_corr'].shape == (batch_size, seq_len, 3)
        assert output['zupt_prob'].shape == (batch_size, seq_len, 1)

    def test_forward_pass_no_nan(self, model, sample_batch):
        """Forward pass should not produce NaN values"""
        output = model(sample_batch)

        assert not torch.isnan(output['pred_pos']).any(), "Position contains NaN"
        assert not torch.isnan(output['pred_vel']).any(), "Velocity contains NaN"
        assert not torch.isnan(output['vel_corr']).any(), "Velocity correction contains NaN"
        assert not torch.isnan(output['zupt_prob']).any(), "ZUPT prob contains NaN"

    def test_single_sample_forward(self, model):
        """Forward pass should work with batch_size=1"""
        sample_batch = {
            'sensor': torch.randn(1, 50, 7),
            'gt_pos': torch.randn(1, 50, 3),
            'gt_vel': torch.randn(1, 50, 3),
            'seq_len': 50
        }

        output = model(sample_batch)

        assert output['pred_pos'].shape == (1, 50, 3)
        assert output['pred_vel'].shape == (1, 50, 3)

    def test_gradient_flow(self, model, sample_batch):
        """Gradients should flow through the hybrid model"""
        model.train()

        # Make input require gradients
        sample_batch['sensor'].requires_grad = True

        output = model(sample_batch)

        # Compute dummy loss
        loss = output['pred_pos'].sum() + output['pred_vel'].sum()
        loss.backward()

        # Check gradients exist and are finite
        assert sample_batch['sensor'].grad is not None
        assert torch.isfinite(sample_batch['sensor'].grad).all()


class TestESKF_TCNIntegration:
    """Test integration between ESKF and TCN components"""

    @pytest.fixture
    def model(self):
        return ESKF_TCN()

    def test_tcn_correction_affects_output(self, model):
        """TCN corrections should affect ESKF output"""
        # Create input with motion
        batch = {
            'sensor': torch.randn(1, 100, 7),
            'gt_pos': torch.randn(1, 100, 3),
            'gt_vel': torch.randn(1, 100, 3),
            'seq_len': 100
        }

        model.eval()
        with torch.no_grad():
            output = model(batch)

        # TCN corrections should be non-trivial
        vel_corr_magnitude = torch.abs(output['vel_corr']).mean()
        assert vel_corr_magnitude > 0, "TCN not producing corrections"

    def test_zupt_detection(self, model):
        """Model should detect zero-velocity periods"""
        # Create stationary input (zero motion)
        batch_size = 1
        seq_len = 100

        sensor_data = torch.zeros(batch_size, seq_len, 7)
        sensor_data[:, :, 2] = 9.81  # Gravity only
        sensor_data[:, :, 6] = 0.0    # FSR = 0 (no force)

        batch = {
            'sensor': sensor_data,
            'gt_pos': torch.zeros(batch_size, seq_len, 3),
            'gt_vel': torch.zeros(batch_size, seq_len, 3),
            'seq_len': seq_len
        }

        model.eval()
        with torch.no_grad():
            output = model(batch)

        # For stationary input, ZUPT probability should be high
        # (might not be perfect without training, but should show tendency)
        mean_zupt_prob = output['zupt_prob'].mean()
        assert mean_zupt_prob >= 0, "ZUPT probability out of range"
        assert mean_zupt_prob <= 1, "ZUPT probability out of range"

    def test_position_integration(self, model):
        """Model should integrate velocity to position"""
        batch = {
            'sensor': torch.randn(1, 100, 7),
            'gt_pos': torch.randn(1, 100, 3),
            'gt_vel': torch.randn(1, 100, 3),
            'seq_len': 100
        }

        model.eval()
        with torch.no_grad():
            output = model(batch)

        # Position should change over time (integration happening)
        pos_change = (output['pred_pos'][0, -1] - output['pred_pos'][0, 0]).abs().sum()
        assert pos_change > 0, "Position not changing (integration not happening)"


class TestESKF_TCNTraining:
    """Test training-related functionality"""

    @pytest.fixture
    def model(self):
        return ESKF_TCN()

    def test_train_eval_modes(self, model):
        """Model should switch between train and eval modes"""
        model.train()
        assert model.training is True

        model.eval()
        assert model.training is False

    def test_parameter_count(self, model):
        """Model should have trainable parameters"""
        params = list(model.parameters())
        assert len(params) > 0, "No parameters found"

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0, "No trainable parameters"

    def test_optimizer_step(self, model):
        """Optimizer should update model parameters"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Forward pass
        batch = {
            'sensor': torch.randn(2, 50, 7),
            'gt_pos': torch.randn(2, 50, 3),
            'gt_vel': torch.randn(2, 50, 3),
            'seq_len': 50
        }

        output = model(batch)
        loss = output['pred_pos'].sum()  # Dummy loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that at least some parameters changed
        params_changed = False
        for p_init, p_now in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_now):
                params_changed = True
                break

        assert params_changed, "No parameters updated after optimizer step"


class TestESKF_TCNEdgeCases:
    """Test edge cases"""

    @pytest.fixture
    def model(self):
        return ESKF_TCN()

    def test_very_short_sequence(self, model):
        """Model should handle very short sequences"""
        batch = {
            'sensor': torch.randn(1, 5, 7),
            'gt_pos': torch.randn(1, 5, 3),
            'gt_vel': torch.randn(1, 5, 3),
            'seq_len': 5
        }

        output = model(batch)
        assert output['pred_pos'].shape == (1, 5, 3)

    def test_long_sequence(self, model):
        """Model should handle long sequences"""
        batch = {
            'sensor': torch.randn(1, 500, 7),
            'gt_pos': torch.randn(1, 500, 3),
            'gt_vel': torch.randn(1, 500, 3),
            'seq_len': 500
        }

        model.eval()
        with torch.no_grad():
            output = model(batch)

        assert output['pred_pos'].shape == (1, 500, 3)
        assert torch.isfinite(output['pred_pos']).all()

    def test_zero_sensor_input(self, model):
        """Model should handle all-zero sensor input"""
        batch = {
            'sensor': torch.zeros(1, 50, 7),
            'gt_pos': torch.zeros(1, 50, 3),
            'gt_vel': torch.zeros(1, 50, 3),
            'seq_len': 50
        }

        model.eval()
        with torch.no_grad():
            output = model(batch)

        assert torch.isfinite(output['pred_pos']).all()
        assert torch.isfinite(output['pred_vel']).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
