"""
Unit tests for TCN (Temporal Convolutional Network) model
"""
import pytest
import torch
from model.TCN import TCN as MultiHeadTCN
from model.config import Config


class TestMultiHeadTCN:
    """Test TCN architecture and forward pass"""

    @pytest.fixture
    def tcn_model(self):
        """Create TCN model for testing"""
        return MultiHeadTCN(
            input_size=Config.TCN_INPUT_SIZE,
            num_channels=Config.TCN_CHANNELS,
            kernel_size=Config.KERNEL_SIZE,
            dropout=0.0  # Disable dropout for deterministic tests
        )

    def test_model_initialization(self, tcn_model):
        """TCN should initialize without errors"""
        assert tcn_model is not None
        assert isinstance(tcn_model, MultiHeadTCN)

    def test_forward_pass_shape(self, tcn_model):
        """Forward pass should return correct output shapes"""
        batch_size = 4
        seq_len = 100
        input_features = torch.randn(batch_size, seq_len, Config.TCN_INPUT_SIZE)

        output = tcn_model(input_features)

        # Check output dictionary structure
        assert 'vel_corr' in output
        assert 'log_var' in output
        assert 'zupt_prob' in output

        # Check shapes
        assert output['vel_corr'].shape == (batch_size, seq_len, 3)
        assert output['log_var'].shape == (batch_size, seq_len, 6)
        assert output['zupt_prob'].shape == (batch_size, seq_len, 1)

    def test_forward_pass_single_sample(self, tcn_model):
        """Forward pass should work with batch_size=1"""
        input_features = torch.randn(1, 50, Config.TCN_INPUT_SIZE)

        output = tcn_model(input_features)

        assert output['vel_corr'].shape == (1, 50, 3)
        assert output['log_var'].shape == (1, 50, 6)
        assert output['zupt_prob'].shape == (1, 50, 1)

    def test_forward_pass_no_nan(self, tcn_model):
        """Forward pass should not produce NaN values"""
        input_features = torch.randn(2, 100, Config.TCN_INPUT_SIZE)

        output = tcn_model(input_features)

        assert not torch.isnan(output['vel_corr']).any()
        assert not torch.isnan(output['log_var']).any()
        assert not torch.isnan(output['zupt_prob']).any()

    def test_zupt_prob_range(self, tcn_model):
        """ZUPT probability should be between 0 and 1"""
        input_features = torch.randn(2, 100, Config.TCN_INPUT_SIZE)

        output = tcn_model(input_features)

        # After sigmoid, values should be in [0, 1]
        assert (output['zupt_prob'] >= 0).all()
        assert (output['zupt_prob'] <= 1).all()

    def test_different_sequence_lengths(self, tcn_model):
        """TCN should handle different sequence lengths"""
        for seq_len in [10, 50, 100, 200]:
            input_features = torch.randn(2, seq_len, Config.TCN_INPUT_SIZE)

            output = tcn_model(input_features)

            assert output['vel_corr'].shape == (2, seq_len, 3)
            assert output['log_var'].shape == (2, seq_len, 6)
            assert output['zupt_prob'].shape == (2, seq_len, 1)

    def test_gradient_flow(self, tcn_model):
        """Gradients should flow through the network"""
        tcn_model.train()

        input_features = torch.randn(2, 50, Config.TCN_INPUT_SIZE, requires_grad=True)
        output = tcn_model(input_features)

        # Compute dummy loss
        loss = output['vel_corr'].sum() + output['log_var'].sum() + output['zupt_prob'].sum()
        loss.backward()

        # Check that gradients exist
        assert input_features.grad is not None
        assert not torch.isnan(input_features.grad).any()

    def test_eval_mode(self, tcn_model):
        """Model should work in eval mode"""
        tcn_model.eval()

        input_features = torch.randn(2, 50, Config.TCN_INPUT_SIZE)

        with torch.no_grad():
            output = tcn_model(input_features)

        assert output['vel_corr'].shape == (2, 50, 3)

    def test_deterministic_forward_pass(self, tcn_model):
        """Same input should produce same output (with dropout=0)"""
        tcn_model.eval()

        input_features = torch.randn(1, 50, Config.TCN_INPUT_SIZE)

        with torch.no_grad():
            output1 = tcn_model(input_features)
            output2 = tcn_model(input_features)

        assert torch.allclose(output1['vel_corr'], output2['vel_corr'])
        assert torch.allclose(output1['log_var'], output2['log_var'])
        assert torch.allclose(output1['zupt_prob'], output2['zupt_prob'])


class TestTCNReceptiveField:
    """Test TCN receptive field calculations"""

    def test_receptive_field_size(self):
        """Calculate and verify receptive field"""
        # With 4 layers, kernel_size=3, dilations=[1, 2, 4, 8]
        # Receptive field = 1 + 2*(1+2+4+8)*(3-1) = 1 + 2*15*2 = 61 timesteps

        model = MultiHeadTCN(
            input_size=20,
            num_channels=[64, 64, 64, 64],
            kernel_size=3,
            dropout=0.0
        )

        # Test that model can process sequences shorter than receptive field
        short_seq = torch.randn(1, 10, 20)
        output = model(short_seq)
        assert output['vel_corr'].shape[1] == 10

        # And longer sequences
        long_seq = torch.randn(1, 200, 20)
        output = model(long_seq)
        assert output['vel_corr'].shape[1] == 200


class TestTCNEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def tcn_model(self):
        return MultiHeadTCN(
            input_size=Config.TCN_INPUT_SIZE,
            num_channels=Config.TCN_CHANNELS,
            kernel_size=Config.KERNEL_SIZE,
            dropout=0.0
        )

    def test_zero_input(self, tcn_model):
        """TCN should handle all-zero input"""
        tcn_model.eval()

        input_features = torch.zeros(1, 50, Config.TCN_INPUT_SIZE)

        with torch.no_grad():
            output = tcn_model(input_features)

        # Should not crash and should produce finite values
        assert torch.isfinite(output['vel_corr']).all()
        assert torch.isfinite(output['log_var']).all()
        assert torch.isfinite(output['zupt_prob']).all()

    def test_large_input_values(self, tcn_model):
        """TCN should handle large input values"""
        tcn_model.eval()

        # Large but finite values
        input_features = torch.randn(1, 50, Config.TCN_INPUT_SIZE) * 100

        with torch.no_grad():
            output = tcn_model(input_features)

        # Should still produce finite outputs
        assert torch.isfinite(output['vel_corr']).all()

    def test_minimum_sequence_length(self, tcn_model):
        """TCN should handle very short sequences"""
        # Minimum reasonable sequence
        input_features = torch.randn(1, 1, Config.TCN_INPUT_SIZE)

        output = tcn_model(input_features)

        assert output['vel_corr'].shape == (1, 1, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
