#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "SatQuery AI - NVIDIA CUDA Acceleration Diagnostics"
echo "============================================================"

if command -v nvidia-smi &>/dev/null; then
    echo "[+] NVIDIA SMI Output:"
    nvidia-smi
else
    echo "[!] Warning: nvidia-smi binary not found."
fi

echo ""
echo "[+] PyTorch CUDA Hardware Verification:"
"$ROOT_DIR/.venv/bin/python" - << 'PYEOF'
import sys
import torch

cuda_ok = torch.cuda.is_available()
print(f"  - PyTorch Version : {torch.__version__}")
print(f"  - CUDA Available  : {cuda_ok}")

if cuda_ok:
    dev_count = torch.cuda.device_count()
    dev_name = torch.cuda.get_device_name(0)
    dev_cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    total_mem_mb = props.total_memory / (1024 * 1024)
    print(f"  - CUDA Devices    : {dev_count}")
    print(f"  - Primary Device  : {dev_name} (Compute Capability {dev_cap[0]}.{dev_cap[1]})")
    print(f"  - Total VRAM      : {total_mem_mb:.1f} MB")
    
    # Simple tensor operation on GPU
    x = torch.randn(512, 512, device='cuda')
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print(f"  - Tensor Compute  : [PASSED] (Allocated: {torch.cuda.memory_allocated(0)/(1024*1024):.2f} MB)")
else:
    print("  - [ERROR] CUDA is not available to PyTorch.")
    sys.exit(1)
PYEOF

echo "============================================================"
