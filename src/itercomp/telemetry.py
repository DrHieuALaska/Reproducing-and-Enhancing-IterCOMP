"""Wall-clock timings and explicit GPU lifecycle, without eager torch imports."""
from contextlib import contextmanager
import gc
import importlib.metadata
import platform
import sys
import time


def synchronize(device=None):
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        if device and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        elif device is None:
            for index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(index)


@contextmanager
def measured(target, key, device=None):
    synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        synchronize(device)
        target[key] = time.perf_counter() - start


def check_device(device, dtype="float32", quantization="none"):
    import torch
    if device == "cpu":
        if quantization != "none" or dtype != "float32":
            raise ValueError("CPU mode requires explicit quantization=none and dtype=float32; no automatic fallback")
        return
    if not str(device).startswith("cuda:"):
        raise ValueError("Specify cpu or an explicit CUDA device such as cuda:0")
    try:
        index = int(device.split(":")[1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid device: {device}") from exc
    if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
        raise RuntimeError(f"{device} unavailable. Enable a notebook GPU or correct device placement; CPU fallback is disabled")
    with torch.cuda.device(index):
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unsupported on this GPU; explicitly select FP16 (recommended for T4)")


def gpu_memory(import_torch=False):
    if import_torch:
        try:
            import torch
        except ImportError:
            return {"cuda_available": False, "devices": [], "reason": "torch not installed"}
    else:
        torch = sys.modules.get("torch")
    if torch is None:
        return {"cuda_available": None, "devices": [], "reason": "torch not imported in this process"}
    if not torch.cuda.is_available():
        return {"cuda_available": False, "devices": []}
    devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        devices.append({"index": index, "name": torch.cuda.get_device_name(index),
                        "free_bytes": free, "total_bytes": total,
                        "allocated_bytes": torch.cuda.memory_allocated(index),
                        "reserved_bytes": torch.cuda.memory_reserved(index),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(index)})
    return {"cuda_available": True, "devices": devices}


def release_memory():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        synchronize()
        torch.cuda.empty_cache()
    return gpu_memory()


def environment_info():
    versions = {}
    for name in ("numpy", "spacy", "PyYAML", "torch", "transformers", "accelerate",
                 "bitsandbytes", "FlagEmbedding", "sentence-transformers", "peft"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "dependencies": versions, "gpu": gpu_memory(), "warmup_policy": "none; first measured sample is cold"}


def sum_known(*values):
    return sum(values) if all(v is not None for v in values) else None
