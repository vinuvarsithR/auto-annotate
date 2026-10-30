"""
check_gpu.py - Phase 0 hardware check.

Run this on EACH machine:
    python scripts/check_gpu.py

It reports:
  1. Python / OS / NVIDIA driver versions
  2. The PyTorch version and which CUDA version it was built with
  3. Every GPU PyTorch can see: name, architecture (sm_XY), VRAM
  4. Whether this PyTorch build has kernels for that architecture
  5. A real test: a small matrix multiply on each GPU

Exit code 0 = at least one GPU works. Exit code 1 = CPU fallback only.
"""

import platform
import shutil
import subprocess
import sys
import time


def line(label: str, value) -> None:
    print(f"  {label:<26} {value}")


def driver_version() -> str:
    """Ask nvidia-smi for the driver version (works on Linux and Windows)."""
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found (driver missing or not on PATH)"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        versions = sorted(set(out.stdout.split()))
        return ", ".join(versions) if versions else "unknown"
    except Exception as exc:  # noqa: BLE001 - report anything, never crash
        return f"could not run nvidia-smi ({exc})"


def test_device(torch, device: str) -> str:
    """Multiply two 2048x2048 matrices on the device. Returns a short result."""
    try:
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        (a @ b).sum().item()  # warm-up
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            c = a @ b
        c.sum().item()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / 10 * 1000
        return f"PASS ({ms:.1f} ms per 2048x2048 matmul)"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL ({type(exc).__name__}: {str(exc).splitlines()[0]})"


def main() -> int:
    print("\n=== System ===")
    line("OS", f"{platform.system()} {platform.release()}")
    line("Python", platform.python_version())
    line("NVIDIA driver", driver_version())

    try:
        import torch
    except ImportError:
        print("\nPyTorch is not installed. See README.md > Setup > Install PyTorch.")
        return 1

    print("\n=== PyTorch ===")
    line("torch version", torch.__version__)
    line("built with CUDA", torch.version.cuda or "none (CPU-only build)")
    line("CUDA available", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("\nNo usable GPU. Falling back to CPU.")
        if torch.version.cuda is None:
            print("Reason: this is a CPU-only PyTorch build. Reinstall with the cu128 command in README.md.")
        line("CPU test", test_device(torch, "cpu"))
        return 1

    arch_list = torch.cuda.get_arch_list()
    line("compiled architectures", " ".join(arch_list))

    print("\n=== GPUs ===")
    any_ok = False
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        major, minor = torch.cuda.get_device_capability(i)
        sm = f"sm_{major}{minor}"
        supported = sm in arch_list
        result = test_device(torch, f"cuda:{i}")
        any_ok = any_ok or result.startswith("PASS")

        print(f"\n  [cuda:{i}] {props.name}")
        line("architecture", sm)
        line("VRAM", f"{props.total_memory / 1024**3:.1f} GB")
        line("kernels in this build", "yes" if supported else "NO - upgrade PyTorch (see README)")
        line("matmul test", result)

    print("\n=== Verdict ===")
    if any_ok:
        print("  GPU OK. This machine can run the models.\n")
        return 0
    print("  No GPU passed the test. The app would fall back to CPU (slow).\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
