"""Lazy SMPL-X and VPoser construction, isolated from input/validation tooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .device import torch_device


@dataclass(frozen=True)
class ModelPaths:
    body_model_path: Path
    vposer_path: Path | None = None
    model_type: str = "smplx"
    gender: str = "neutral"
    num_betas: int = 10


def load_body_model(paths: ModelPaths, device: str = "auto"):
    """Create SMPL/SMPL-H/SMPL-X only when fitting is explicitly requested."""
    try:
        import smplx
    except ImportError as error:
        raise RuntimeError("install fitting extras first: python -m pip install -e '.[fit]'") from error
    if not paths.body_model_path.exists():
        raise FileNotFoundError(f"SMPL body model directory does not exist: {paths.body_model_path}")
    model_type = paths.model_type.lower()
    direct_model_file = paths.body_model_path / f"{model_type.upper()}_{paths.gender.upper()}.npz"
    # smplx.create(root, model_type="smplx") expects root/smplx/*.npz. The
    # project intentionally stores models directly in models/smplx/*.npz, so
    # instantiate the class directly for that more convenient layout.
    if direct_model_file.exists():
        model_class = {"smpl": smplx.SMPL, "smplh": smplx.SMPLH, "smplx": smplx.SMPLX}.get(model_type)
        if model_class is None:
            raise ValueError(f"unsupported body model type: {paths.model_type!r}")
        model = model_class(
            str(paths.body_model_path), gender=paths.gender, num_betas=paths.num_betas,
            use_pca=False, batch_size=1,
        )
    else:
        model = smplx.create(
            str(paths.body_model_path), model_type=model_type, gender=paths.gender,
            num_betas=paths.num_betas, use_pca=False, batch_size=1,
        )
    return model.to(torch_device(device))


def load_vposer(paths: ModelPaths, device: str = "auto"):
    if paths.vposer_path is None:
        return None
    from .vposer import load_vposer_v02_05

    return load_vposer_v02_05(paths.vposer_path, torch_device(device))
