"""Safe PyTorch device selection for macOS, CUDA hosts and CPU-only machines."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    selected: str
    torch_version: str
    platform: str
    mps_built: bool
    mps_available: bool
    cuda_available: bool

    def to_mapping(self) -> dict[str, str | bool]:
        return asdict(self)


def select_device(requested: str = "auto") -> DeviceInfo:
    """Select a valid accelerator without ever assuming CUDA on a Mac.

    On Apple Silicon `mps` is selected only after PyTorch reports it usable.
    `PYTORCH_ENABLE_MPS_FALLBACK` cannot fix an unavailable MPS device, so we
    intentionally do not set it here.
    """
    try:
        import platform
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required: python -m pip install -e '.[fit]'") from error

    requested = requested.lower()
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, mps, cuda")
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        selected = "mps" if mps_available else "cuda" if cuda_available else "cpu"
    elif requested == "mps" and not mps_available:
        raise RuntimeError("MPS was explicitly requested but is not available in this PyTorch/macOS session")
    elif requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was explicitly requested but is not available")
    else:
        selected = requested
    return DeviceInfo(
        requested=requested, selected=selected, torch_version=torch.__version__,
        platform=platform.platform(), mps_built=mps_built, mps_available=mps_available,
        cuda_available=cuda_available,
    )


def torch_device(requested: str = "auto"):
    """Return the selected torch.device, keeping imports out of data-only modules."""
    import torch

    return torch.device(select_device(requested).selected)
