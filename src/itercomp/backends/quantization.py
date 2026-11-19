"""Validate the native quantization backend without downloading/loading weights."""
import importlib.metadata


def check_quantization_runtime(config):
    if config.backend != "transformers" or config.quantization == "none":
        return None
    import torch
    from packaging.version import Version
    repair = ('Reinstall in the notebook interpreter: python -m pip install --no-cache-dir '
              '--force-reinstall --no-deps bitsandbytes==0.49.2\n'
              'Then restart the kernel. Ensure requirements-gpu.txt also pins 0.49.2; '
              'an older working copy can reinstall an incompatible build.')
    if not config.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("Quantized Transformers inference requires the configured CUDA GPU; no CPU fallback is performed")
    try:
        version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError("bitsandbytes is not installed. " + repair) from None
    cuda = torch.version.cuda
    if not cuda:
        raise RuntimeError("PyTorch does not report a CUDA build; enable the platform CUDA runtime")
    if Version(cuda) >= Version("13.0") and Version(version) < Version("0.49.2"):
        raise RuntimeError(f"bitsandbytes {version} is older than the selected CUDA-13-compatible pin, while PyTorch uses CUDA {cuda}. " + repair)
    try:
        from bitsandbytes.cextension import lib
    except Exception as exc:
        raise RuntimeError(f"bitsandbytes {version} native backend failed to import for CUDA {cuda}. " + repair) from exc
    if lib is None or getattr(lib, "compiled_with_cuda", False) is not True:
        raise RuntimeError(f"bitsandbytes {version} did not load its native CUDA backend (PyTorch CUDA {cuda}); NF4/INT8 GPU loading cannot proceed. " + repair)
    return {"bitsandbytes_version": version, "torch_cuda": cuda, "native_cuda_backend": True}
