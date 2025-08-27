# GPU Keepalive

**RTX P-State Management for Local AI (Research Preview)**

GPU Keepalive is a lightweight Python tool for **testing RTX GPU performance** during local AI inference. It is designed for **research, experimentation, and demos**, not production use.

**The Problem**: RTX GPUs automatically drop from high-performance P0 state to power-saving P8 state between AI inference requests, causing dramatic slowdowns (up to 4x performance loss).

**The Solution**: Keep your RTX GPU in optimal P0 state through minimal background operations.

## Research Background

Read more about this tool as part of the broader Artificial Mind framework: [The Artificial Mind Papers – Section 0: The Intro](https://medium.com/@luislozanog86/the-artificial-mind-papers-section-0-the-intro-how-validation-before-action-changes-everything-becfeb3a0ddc)

## Key Features (Research Use)

* **P-State Management (experimental)**: Maintains RTX GPUs in P0 state during idle periods (max performance state)
* **Thermal Protection**: Automatically pauses during high temperatures
* **Memory Safe**: Minimal VRAM usage (~130KB) with proper cleanup
* **Zero Dependencies**: Works with any PyTorch-based LLM setup
* **Hardware Tested**: Validated on RTX 3070 Laptop (85W)

**Note**: This is for **testing only**. Validate thoroughly on your hardware before any extended use.

## Quick Start (Demo)

```python
from gpu_keepalive import GPUKeepalive

# Start keepalive
keepalive = GPUKeepalive(debug=True)
keepalive.start()

# Your existing LLM code
# model = AutoModelForCausalLM.from_pretrained("your-model")
# GPU stays in P0 automatically

# Stop when done
keepalive.stop()
```

## Installation

```bash
git clone https://github.com/luislozanogmia/gpu-keepalive.git
cd gpu-keepalive
python setup.py
```

The setup script will analyze your GPU and recommend optimal models for your VRAM.

## How It Works

GPU Keepalive prevents P-state drops through tiny background operations:

1. **Background Thread**: Runs micro-operations (matrix multiply) every 100ms
2. **Smart Pausing**: Automatically stops during active inference
3. **Thermal Monitoring**: Pauses if GPU temperature gets too high
4. **Resource Management**: Single persistent tensor, immediate cleanup

## Validation

**Check P-State Status:**
```bash
# Monitor GPU state in real-time
nvidia-smi -l 1
# Look for "P0" instead of "P8" in power state column
```

## GPU Recommendations (Research Data)

Based on testing, here are model recommendations for different VRAM sizes:

| VRAM | Conservative Choice | Aggressive Choice |
|------|-------------------|------------------|
| 6GB  | 3B (4-bit quantized) | - |
| 8GB  | 3B (4-bit) | 3B (full precision) |
| 12GB | 7B (4-bit) | 12B (4-bit) |
| 16GB | 12B (4-bit) | 20B (4-bit) |
| 24GB+ | 20B (4-bit) | 70B+ (4-bit) |

*These are research observations. Your results may vary.*

## Context Manager Usage

```python
keepalive = GPUKeepalive()
keepalive.start()

# Pause during active inference
with keepalive.inference_mode():
    outputs = model.generate(inputs)
# Automatically resumes background keepalive
```

## Requirements

* **GPU**: NVIDIA RTX series (2080, 3070, 4090, etc.)
* **CUDA**: Compatible drivers installed  
* **Python**: 3.8+ with PyTorch
* **Dependencies**: Auto-installed by setup.py

## Research Validation (Internal)

Tested on RTX 3070 Laptop (85W):
* P-state consistency: P0 maintained during idle periods
* Thermal behavior: Automatic pause above 83°C
* Memory usage: ~130KB persistent VRAM
* Performance: Eliminates P8 state transitions

*These results are from internal testing. Validate on your hardware.*

## Troubleshooting

**Still seeing P8 states**: Check for other GPU processes with nvidia-smi, try increasing keepalive frequency, or verify no other applications are managing GPU clocks
**Temperature warnings**: GPU is running hot, thermal protection is working correctly. Improve ventilation or lower the max temperature threshold
**High performance**: Make sure your computer is set to high or ultimate performance mode and plugged in (critical for laptops)
**Memory errors**: If you get CUDA out of memory errors, the keepalive tensor might conflict with model loading. Try reducing tensor_size or restarting Python
**Multiple GPUs**: Specify the correct device_id if you have multiple GPUs: GPUKeepalive(device_id=1)
**Driver issues**: Update to latest NVIDIA drivers if you experience crashes or unexpected behavior
**Keepalive not working**: Verify the background thread is running with debug=True and check you're not accidentally calling stop() too early

## Related Research Tools

This tool is part of a broader ecosystem for local AI experimentation:

* **[ContextZip](https://github.com/luislozanogmia/contextzip)**: Semantic context compression (50-90% token reduction)
* **[macOS Accessibility Fix](https://github.com/luislozanogmia/macos-electron-accessibility-fix)**: macOS accessibility improvements for AI tools

## Testing & Feedback

Found RTX P-state issues this doesn't solve? Have performance data from different GPU models?

* **Issues**: Report specific RTX models and performance patterns
* **Discussions**: Share your GPU testing results
* **Contributions**: Additional GPU architectures and thermal improvements welcome

## Disclaimer

**Research Tool**: Tested primarily on RTX 3070 Laptop (85W). Use for research, demonstrations, and experimentation. Validate thoroughly on your hardware configuration.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Stop fighting RTX power management. Start building reliable local AI.**
