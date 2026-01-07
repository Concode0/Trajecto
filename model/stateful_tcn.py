import torch
import torch.nn as nn
from model.TCN import TCN
from model.config import Config

class StatefulTCNExport(nn.Module):
    """
    Wrapper around a trained TCN model to support stateful (step-by-step) execution.
    Designed primarily for ONNX export to embedded systems.
    
    Instead of recomputing the entire receptive field at each step, this model
    takes the current input and a list of state buffers (previous inputs to each layer),
    and returns the prediction and the updated state buffers.
    
    Layout: Standard PyTorch NCL [Batch, Channels, Length].
    """
    def __init__(self, tcn_model: TCN):
        super().__init__()
        self.layers = tcn_model.tcn_layers
        self.output_heads = tcn_model.output_heads

        # No input normalization - features are pre-normalized upstream
        # This preserves physical relationships between feature magnitudes

        # Precompute buffer sizes for each layer
        self.buffer_sizes = []
        for layer in self.layers:
            # layer is CausalConv1d
            if layer.separable:
                # Use depthwise conv params
                k = layer.depthwise.kernel_size[0]
                d = layer.depthwise.dilation[0]
            else:
                # Use standard conv params
                k = layer.conv.kernel_size[0]
                d = layer.conv.dilation[0]
            
            # History needed is (Kernel-1) * Dilation
            self.buffer_sizes.append((k - 1) * d)

    def forward(self, x_t, *states):
        """
        Args:
            x_t: Current input feature [Batch, 1, InputSize] -> Transpose to [Batch, InputSize, 1]
            *states: Variable number of state tensors. 
                     Expected len(states) == len(layers).
                     State i shape: [Batch, InChannels_i, BufferSize_i] (NCL)
        
        Returns:
            outputs (tuple): (vel_corr, covariance_R, zupt_prob) each [Batch, 1, OutDim]
            new_states (tuple): Updated state tensors.
        """
        # Input x_t is [Batch, 1, C] (NLC) from external world.
        # Transpose to [Batch, C, 1] (NCL)
        # Features are already normalized upstream (see base_hybrid_model.py)
        current_input = x_t.transpose(1, 2)
        
        new_states = []
        
        for i, layer in enumerate(self.layers):
            buffer = states[i] # [Batch, C, Hist]
            
            # Concatenate history with current input on Time (dim 2)
            # input_window: [Batch, C, Hist + 1]
            input_window = torch.cat([buffer, current_input], dim=2)
            
            # Perform Convolution
            # nn.Conv1d expects [Batch, C, Length] - no transpose needed here
            
            if layer.separable:
                # Depthwise (Space)
                out = layer.depthwise(input_window) # [Batch, C, 1]
                # Pointwise (Cross-Channel)
                out = layer.pointwise(out)          # [Batch, OutC, 1]
            else:
                out = layer.conv(input_window)      # [Batch, OutC, 1]
            
            out = layer.relu(out)
            out = layer.dropout(out)
            
            # Update State: The new buffer is the shifted input window
            # We discard the oldest sample (index 0 in time dim 2) and keep the rest
            # new_buffer: [Batch, C, Hist]
            new_buffer = input_window[:, :, 1:]
            new_states.append(new_buffer)
            
            # Output becomes input for next layer
            # out is [Batch, OutC, 1] - already NCL
            current_input = out
            
        # Final Heads
        # current_input: [Batch, FinalC, 1] -> Transpose to [Batch, 1, FinalC] for linear layers
        final_feature = current_input.transpose(1, 2)

        # Return outputs in a fixed order for ONNX
        vel_corr = self.output_heads["vel_corr"](final_feature)
        # Apply tanh + physics scale (matching TCN.py forward pass)
        vel_corr = torch.tanh(vel_corr) * Config.VEL_CORRECTION_SCALE
        # Hard-clip velocity correction to prevent extreme values
        vel_corr = torch.clamp(
            vel_corr,
            min=-Config.ESKFTCN.VEL_CORR_CLIP_RANGE,
            max=Config.ESKFTCN.VEL_CORR_CLIP_RANGE
        )
        cov_R = self.output_heads["covariance_R"](final_feature)
        # CRITICAL: Clip covariance output to prevent numerical explosion
        # Clamping to [-10, 5] → variance range after softplus: [~4.5e-5, ~150]
        cov_R = torch.clamp(
            cov_R,
            min=-10.0,
            max=5.0
        )
        zupt_p = self.output_heads["zupt_prob"](final_feature)

        return (vel_corr, cov_R, zupt_p), tuple(new_states)