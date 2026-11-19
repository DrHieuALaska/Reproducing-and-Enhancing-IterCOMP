"""Small notebook helpers. Nothing is downloaded by importing this module."""
from dataclasses import asdict
from pathlib import Path
import platform
from .config import load_config


def configure_hf_auth(platform_name, online=True):
    """Use an existing login or notebook secret; never print/store the credential."""
    import os
    from huggingface_hub import HfApi, get_token
    if platform_name not in ("kaggle", "colab"):
        raise ValueError("platform_name must be kaggle or colab")
    token = get_token()
    source = "existing environment or cached Hugging Face login" if token else None
    secret_error = None
    if not token:
        try:
            if platform_name == "kaggle":
                from kaggle_secrets import UserSecretsClient
                token = UserSecretsClient().get_secret("HF_TOKEN")
            else:
                from google.colab import userdata
                token = userdata.get("HF_TOKEN")
            token = token.strip() if isinstance(token, str) else None
            if token:
                source = f"{platform_name} HF_TOKEN secret"
        except Exception as exc:
            # Some exception messages can contain sensitive details; report only type.
            secret_error = type(exc).__name__
    if not token:
        location = "Kaggle Add-ons > Secrets" if platform_name == "kaggle" else "Colab's Secrets panel"
        print(f"No HF_TOKEN or saved login found. Add a Hugging Face read token named HF_TOKEN in {location}, enable notebook access, then rerun this cell. Public models or complete attached checkpoints may work without a token.")
        return {"authenticated": False, "source": None, "secret_error": secret_error}
    # Export cached credentials too, so the separate compression/reader processes
    # inherit the token even if their cache locations differ from the notebook's.
    os.environ["HF_TOKEN"] = token
    if online:
        try:
            HfApi(token=token).whoami()
        except Exception:
            raise RuntimeError("HF_TOKEN was found, but Hugging Face authentication could not be validated. Check internet access and token validity. Update the HF_TOKEN environment variable, or restart the kernel after changing the notebook secret.") from None
        print("Hugging Face authentication verified. Gated model access must also be approved for this account.")
    else:
        print("HF_TOKEN found; online validation skipped because INTERNET is False.")
    return {"authenticated": bool(online), "token_available": True, "source": source}


def notebook_config(profile_path, output_path, checkpoints=None, revisions=None,
                    internet=True, encoder_device="cuda:0", controller_device="cuda:0", reader_device="cuda:0"):
    """Resolve one profile for an attached checkpoint or explicit cloud download."""
    overrides = {}
    devices = {"embedding": encoder_device, "controller": controller_device, "reader": reader_device}
    for name in ("embedding", "controller", "reader"):
        overrides[name] = {"device": devices[name], "local_files_only": not internet}
        if checkpoints and checkpoints.get(name):
            overrides[name]["model_id"] = str(Path(checkpoints[name]).resolve())
        if revisions and revisions.get(name):
            overrides[name]["revision"] = revisions[name]
    config = load_config(profile_path, overrides)
    import yaml
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    return config


def verify_model_access(config, roles=("embedding", "controller", "reader")):
    """Check metadata/local files only; never downloads checkpoint weights."""
    from huggingface_hub import HfApi
    results = []
    for role in roles:
        model = getattr(config, role)
        if model.backend in ("mock", "openai_compatible"):
            results.append({"role": role, "backend": model.backend, "verification": "not an HF checkpoint"})
            continue
        path = Path(model.model_id)
        if path.is_dir():
            if not (path / "config.json").exists():
                raise FileNotFoundError(f"{role}: no config.json under attached checkpoint {path}")
            if role == "embedding" and not (path / "sparse_linear.pt").is_file():
                raise FileNotFoundError("Attached BGE-M3 checkpoint must include sparse_linear.pt")
            results.append({"role": role, "path": str(path), "verification": "local config/head presence only"})
        elif model.local_files_only:
            raise ValueError(f"{role}: offline execution requires an attached local checkpoint path; configure CHECKPOINTS")
        else:
            info = HfApi().model_info(model.model_id, revision=model.revision)
            results.append({"role": role, "model_id": model.model_id, "revision": info.sha,
                            "verification": "Hub metadata only; gated weight access still checked at loading"})
    return results


def runtime_check():
    from .telemetry import gpu_memory
    import importlib.metadata
    from packaging.version import Version
    version = Version(importlib.metadata.version("torch"))
    if not Version("2.6") <= version < Version("2.8"):
        print(f"PyTorch {version}: intended stack is 2.6.x/2.7.x; this platform version needs separate validation. It has not been replaced.")
    if not (3, 10) <= tuple(int(x) for x in platform.python_version_tuple()[:2]) <= (3, 12):
        print("The GPU dependency pins target Python 3.10-3.12; validate this runtime before model loading.")
    memory = gpu_memory(import_torch=True)
    if not memory["cuda_available"]:
        raise RuntimeError("Enable a Kaggle/Colab GPU runtime. This notebook does not fall back to CPU inference")
    return memory
