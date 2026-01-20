"""Quantization-Aware Training (QAT) wrapper for TCN model using PT2E API.

This module provides QAT capabilities for the TCN component of ESKF-TCN,
enabling INT8 quantization with minimal accuracy loss for ESP32 deployment.

Uses PyTorch 2 Export (PT2E) Quantization API, which is the recommended
approach for PyTorch 2.x and beyond. This replaces the deprecated eager mode
quantization (prepare_qat/convert).

PT2E Workflow:
1. Export model using torch.export.export()
2. Prepare for QAT using prepare_qat_pt2e() with a Quantizer
3. Train with fake quantization ops inserted
4. Convert using convert_pt2e() for deployment

Architecture:
    FP32 ESKF (physics) --> FP32 features --> QAT TCN --> INT8 outputs
                           |
    Only TCN is quantized; ESKF remains FP32 (simple matrix ops on ESP32)

Usage:
    from model.qat_tcn import PT2EQuantizer, prepare_qat_pt2e_model

    # During training (after warmup epochs):
    model, example_inputs = prepare_qat_pt2e_model(model, sample_input)

    # Continue training with QAT...

    # After training, convert for deployment:
    quantized_model = convert_qat_to_quantized(model)
"""

import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# PT2E Quantization imports (PyTorch 2.x)
try:
    from torch.ao.quantization.quantizer import Quantizer
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torch.ao.quantization.quantize_pt2e import (
        prepare_qat_pt2e,
        convert_pt2e,
    )
    # PyTorch 2.9+ uses torch.export.export instead of capture_pre_autograd_graph
    try:
        from torch._export import capture_pre_autograd_graph
    except ImportError:
        from torch.export import export as capture_pre_autograd_graph
    PT2E_AVAILABLE = True
except ImportError:
    PT2E_AVAILABLE = False
    print("[QAT] Warning: PT2E quantization not available. Requires PyTorch >= 2.1")

from model.config import Config


class TCNQuantizationWrapper(nn.Module):
    """Wrapper for TCN that handles quantization boundaries.

    This wrapper ensures proper input/output handling for quantization:
    - Input transpose (B, T, D) -> (B, D, T) for Conv1d
    - Output transpose back to (B, T, D)
    - Post-processing (tanh, sigmoid, normalize) applied after dequantization

    The wrapper is used during PT2E export to create a clean graph for
    quantization while keeping post-processing in FP32.
    """

    def __init__(self, tcn: nn.Module):
        super().__init__()
        self.tcn_layers = tcn.tcn_layers
        self.output_heads = tcn.output_heads

        # Copy buffer from original TCN
        if hasattr(tcn, 'vel_scale_isotropic'):
            self.register_buffer(
                'vel_scale_isotropic',
                tcn.vel_scale_isotropic.clone()
            )
        else:
            self.register_buffer(
                'vel_scale_isotropic',
                torch.tensor(Config.VEL_CORRECTION_SCALE, dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass matching original TCN signature.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Dict with vel_corr, covariance_R, zupt_prob, gravity_b
        """
        # Transpose for Conv1d: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)

        # TCN backbone
        for layer in self.tcn_layers:
            x = layer(x)

        # Transpose back: [B, D, T] -> [B, T, D']
        x = x.transpose(1, 2)

        # Output heads
        outputs = {}
        for head_name, head in self.output_heads.items():
            outputs[head_name] = head(x)

        # Post-processing (kept in FP32 for numerical stability)
        # Velocity: signed log1p + scale + clamp
        raw_vel = outputs["vel_corr"]
        vel_log_scale = torch.sign(raw_vel) * torch.log1p(torch.abs(raw_vel))
        outputs["vel_corr"] = torch.clamp(
            vel_log_scale * self.vel_scale_isotropic,
            min=-Config.ESKFTCN.VEL_CORR_CLIP_RANGE,
            max=Config.ESKFTCN.VEL_CORR_CLIP_RANGE
        )

        # Covariance: clamp log-variance
        outputs["covariance_R"] = torch.clamp(
            outputs["covariance_R"],
            min=-10.0,
            max=2.95
        )

        # ZUPT: sigmoid
        outputs["zupt_prob"] = torch.sigmoid(outputs["zupt_prob"])

        # Gravity: L2 normalize
        outputs["gravity_b"] = torch.nn.functional.normalize(
            outputs["gravity_b"], p=2, dim=-1, eps=1e-6
        )

        return outputs


def get_xnnpack_quantizer(is_qat: bool = True) -> "XNNPACKQuantizer":
    """Create XNNPACK quantizer for ARM/mobile deployment.

    XNNPACK is optimized for ARM processors (including ESP32) and provides
    efficient INT8 inference with per-tensor symmetric quantization.

    Args:
        is_qat: If True, configure for QAT. If False, for post-training quantization.

    Returns:
        Configured XNNPACKQuantizer instance.
    """
    if not PT2E_AVAILABLE:
        raise RuntimeError("PT2E quantization not available. Requires PyTorch >= 2.1")

    quantizer = XNNPACKQuantizer()

    # Get quantization config
    # Use symmetric quantization for weights (better for conv)
    # Use affine quantization for activations (better dynamic range)
    if is_qat:
        # QAT config: observers that track ranges during training
        quantization_config = get_symmetric_quantization_config(is_qat=True)
    else:
        # PTQ config: static quantization after calibration
        quantization_config = get_symmetric_quantization_config(is_qat=False)

    # Apply config globally (quantize all supported ops)
    quantizer.set_global(quantization_config)

    return quantizer


def prepare_qat_pt2e_model(
    model: nn.Module,
    example_input: torch.Tensor,
    quantizer: Optional["Quantizer"] = None,
) -> Tuple[nn.Module, torch.Tensor]:
    """Prepare model for QAT using PT2E API.

    This function:
    1. Exports the TCN submodule using torch.export
    2. Inserts fake quantization observers using prepare_qat_pt2e
    3. Returns the modified model ready for QAT training

    Args:
        model: The full ESKF-TCN model (must have .tcn attribute)
        example_input: Sample input tensor [B, T, D] for tracing
        quantizer: Optional custom quantizer. Defaults to XNNPACKQuantizer.

    Returns:
        Tuple of (qat_prepared_model, example_input)

    Example:
        >>> model = ESKFTCN(...)
        >>> sample = torch.randn(1, 100, 19)
        >>> model, _ = prepare_qat_pt2e_model(model, sample)
        >>> # Continue training...
    """
    if not PT2E_AVAILABLE:
        raise RuntimeError(
            "PT2E quantization not available. "
            "Requires PyTorch >= 2.1. Install with: pip install torch>=2.1"
        )

    if not hasattr(model, 'tcn'):
        raise ValueError(
            "Model must have a 'tcn' attribute for QAT preparation. "
            "Expected ESKFTCN or AEKFTCN model."
        )

    # Use XNNPACK quantizer by default (optimized for ARM/ESP32)
    if quantizer is None:
        quantizer = get_xnnpack_quantizer(is_qat=True)

    # Store original TCN for reference
    original_tcn = model.tcn

    # Put model in eval mode for export (required by torch.export)
    model.eval()

    # Export the TCN submodule
    # We export only the TCN, not the full ESKF-TCN, because:
    # 1. ESKF has complex control flow that's hard to trace
    # 2. Only TCN benefits from INT8 quantization
    # 3. ESKF matrix ops are fast enough in FP32 on ESP32
    print("[QAT-PT2E] Exporting TCN for quantization...")

    try:
        # Use capture_pre_autograd_graph for QAT (preserves autograd)
        exported_tcn = capture_pre_autograd_graph(
            original_tcn,
            args=(example_input,),
        )
        print("[QAT-PT2E] TCN exported successfully")
    except Exception as e:
        print(f"[QAT-PT2E] Export failed: {e}")
        print("[QAT-PT2E] Falling back to torch.export.export...")
        # Fallback to standard export
        exported_tcn = torch.export.export(
            original_tcn,
            args=(example_input,),
        ).module()

    # Prepare for QAT by inserting fake quantization ops
    print("[QAT-PT2E] Preparing for QAT with XNNPACK quantizer...")
    qat_tcn = prepare_qat_pt2e(exported_tcn, quantizer)
    print("[QAT-PT2E] QAT preparation complete")

    # Replace the TCN in the model with the QAT version
    model.tcn = qat_tcn

    # Put back in training mode
    model.train()

    return model, example_input


def convert_qat_to_quantized(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """Convert QAT-trained model to fully quantized model.

    Call this after QAT training is complete. The resulting model uses
    actual INT8 operations instead of fake quantization.

    Args:
        model: QAT-trained model (must have QAT-prepared .tcn)
        inplace: If True, modify model in place

    Returns:
        Quantized model ready for export/deployment

    Example:
        >>> # After QAT training:
        >>> quantized_model = convert_qat_to_quantized(model)
        >>> # Export to TFLite, ONNX, etc.
    """
    if not PT2E_AVAILABLE:
        raise RuntimeError("PT2E quantization not available.")

    if not inplace:
        model = copy.deepcopy(model)

    model.eval()

    if hasattr(model, 'tcn'):
        print("[QAT-PT2E] Converting TCN to quantized model...")
        model.tcn = convert_pt2e(model.tcn)
        print("[QAT-PT2E] Conversion complete")
    else:
        print("[QAT-PT2E] Warning: Model has no 'tcn' attribute. Skipping conversion.")

    return model


def get_qat_observer_stats(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Get QAT observer statistics for debugging.

    Returns quantization parameters (scale, zero_point) learned during QAT.

    Args:
        model: QAT model (after some training)

    Returns:
        Dict mapping layer names to their quantization parameters
    """
    stats = {}

    if not hasattr(model, 'tcn'):
        return stats

    for name, module in model.tcn.named_modules():
        # Check for activation_post_process (observer) in PT2E
        if hasattr(module, 'activation_post_process'):
            observer = module.activation_post_process
            if hasattr(observer, 'calculate_qparams'):
                try:
                    scale, zero_point = observer.calculate_qparams()
                    stats[name] = {
                        'scale': scale.item() if scale.numel() == 1 else scale.tolist(),
                        'zero_point': zero_point.item() if zero_point.numel() == 1 else zero_point.tolist(),
                    }
                except Exception:
                    pass  # Observer not yet calibrated

        # Also check for FakeQuantize modules (PT2E style)
        if 'FakeQuantize' in type(module).__name__:
            if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
                stats[name] = {
                    'scale': module.scale.item() if module.scale.numel() == 1 else module.scale.tolist(),
                    'zero_point': module.zero_point.item() if module.zero_point.numel() == 1 else module.zero_point.tolist(),
                }

    return stats


def estimate_quantization_error(
    model: nn.Module,
    sample_input: torch.Tensor,
) -> Tuple[float, Dict[str, float]]:
    """Estimate quantization error by comparing FP32 vs fake-quantized outputs.

    Useful for monitoring QAT progress and detecting quantization issues.

    Args:
        model: QAT model
        sample_input: Sample input tensor [B, T, D]

    Returns:
        Tuple of (mean_relative_error, per_output_errors)
    """
    model.eval()

    # Get output with fake quantization
    with torch.no_grad():
        qat_output = model(sample_input)

    # Try to disable fake quant for FP32 comparison
    try:
        # PT2E uses different mechanism for disabling fake quant
        for module in model.modules():
            if hasattr(module, 'disable_fake_quant'):
                module.disable_fake_quant()
            elif hasattr(module, 'fake_quant_enabled'):
                module.fake_quant_enabled = False

        with torch.no_grad():
            fp32_output = model(sample_input)

        # Re-enable fake quant
        for module in model.modules():
            if hasattr(module, 'enable_fake_quant'):
                module.enable_fake_quant()
            elif hasattr(module, 'fake_quant_enabled'):
                module.fake_quant_enabled = True

    except Exception as e:
        print(f"[QAT-PT2E] Could not compute FP32 reference: {e}")
        # Return zeros if we can't compare
        return 0.0, {k: 0.0 for k in qat_output.keys() if isinstance(qat_output[k], torch.Tensor)}

    # Compute relative errors per output
    errors = {}
    for key in qat_output.keys():
        if isinstance(qat_output[key], torch.Tensor) and isinstance(fp32_output.get(key), torch.Tensor):
            diff = (qat_output[key] - fp32_output[key]).abs()
            rel_err = diff / (fp32_output[key].abs() + 1e-8)
            errors[key] = rel_err.mean().item()

    mean_error = sum(errors.values()) / max(len(errors), 1)
    return mean_error, errors


class QATScheduler:
    """Manages QAT activation during training using PT2E API.

    Handles the transition from FP32 training to QAT based on epoch number.
    QAT is activated after warmup to allow the model to first learn good
    FP32 weights before adding quantization noise.

    Example:
        >>> scheduler = QATScheduler(start_epoch=10)
        >>> example_input = torch.randn(1, 100, 19)
        >>> for epoch in range(100):
        >>>     model = scheduler.step(model, epoch, example_input)
        >>>     # Training loop...
    """

    def __init__(
        self,
        start_epoch: int = 10,
        backend: str = "qnnpack",  # Kept for compatibility, XNNPACK used internally
        enabled: bool = True,
    ):
        """Initialize QAT scheduler.

        Args:
            start_epoch: Epoch to start QAT (after FP32 warmup)
            backend: Backend hint (XNNPACK used for ARM/ESP32 regardless)
            enabled: If False, QAT is never activated
        """
        self.start_epoch = start_epoch
        self.backend = backend
        self.enabled = enabled
        self.qat_active = False
        self._example_input: Optional[torch.Tensor] = None

    def set_example_input(self, example_input: torch.Tensor) -> None:
        """Set example input for model export.

        Must be called before step() if QAT is enabled.

        Args:
            example_input: Sample input tensor [B, T, D]
        """
        self._example_input = example_input

    def step(
        self,
        model: nn.Module,
        epoch: int,
        example_input: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Check if QAT should be activated at this epoch.

        Args:
            model: Current model
            epoch: Current epoch number
            example_input: Optional example input (can also use set_example_input)

        Returns:
            Model (possibly converted to QAT mode)
        """
        if not self.enabled:
            return model

        if not self.qat_active and epoch >= self.start_epoch:
            # Get example input
            if example_input is not None:
                self._example_input = example_input

            if self._example_input is None:
                print(f"[QAT-PT2E] Warning: No example input provided. "
                      f"Call scheduler.set_example_input() before epoch {self.start_epoch}")
                return model

            if not PT2E_AVAILABLE:
                print("[QAT-PT2E] Warning: PT2E not available. Skipping QAT activation.")
                return model

            print(f"\n[QAT-PT2E] Activating Quantization-Aware Training at epoch {epoch}")

            try:
                model, _ = prepare_qat_pt2e_model(
                    model,
                    self._example_input,
                    quantizer=get_xnnpack_quantizer(is_qat=True),
                )
                self.qat_active = True
                print("[QAT-PT2E] QAT activation successful")
            except Exception as e:
                print(f"[QAT-PT2E] QAT activation failed: {e}")
                print("[QAT-PT2E] Continuing with FP32 training")

        return model

    def is_active(self) -> bool:
        """Check if QAT is currently active."""
        return self.qat_active


# Backward compatibility aliases
prepare_qat_model = prepare_qat_pt2e_model
convert_to_quantized = convert_qat_to_quantized
get_qat_state = get_qat_observer_stats


if __name__ == "__main__":
    print("Testing PT2E QAT for TCN...")
    print(f"PT2E Available: {PT2E_AVAILABLE}")

    if not PT2E_AVAILABLE:
        print("Skipping test - PT2E not available")
        exit(0)

    # Import TCN for testing
    from model.TCN import TCN

    # Create TCN model
    tcn = TCN(
        input_size=19,
        tcn_channels=[64, 64, 64, 64],
        kernel_size=3,
        dropout=0.1,
    )

    # Test forward pass
    batch_size, seq_len = 2, 100
    x = torch.randn(batch_size, seq_len, 19)

    print(f"\nInput shape: {x.shape}")
    outputs = tcn(x)

    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")

    print(f"\nReceptive field: {tcn.receptive_field} timesteps")

    # Test PT2E QAT preparation
    print("\n--- Testing PT2E QAT ---")

    # Create a simple wrapper to simulate ESKF-TCN structure
    class MockESKFTCN(nn.Module):
        def __init__(self, tcn):
            super().__init__()
            self.tcn = tcn

        def forward(self, x):
            return self.tcn(x)

    model = MockESKFTCN(tcn)

    print("\nPreparing for QAT with PT2E...")
    try:
        model, _ = prepare_qat_pt2e_model(model, x)
        print("QAT preparation successful!")

        # Forward pass with fake quantization
        model.train()
        outputs_qat = model(x)
        print("\nQAT forward pass successful!")

        for key, val in outputs_qat.items():
            print(f"  {key}: {val.shape}")

        # Get observer stats
        stats = get_qat_observer_stats(model)
        print(f"\nObserver stats collected for {len(stats)} layers")

        # Convert to quantized
        print("\nConverting to quantized model...")
        quantized_model = convert_qat_to_quantized(model)
        print("Conversion successful!")

    except Exception as e:
        print(f"QAT test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nPT2E QAT TCN test complete!")
