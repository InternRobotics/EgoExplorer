"""Minimal inference-only implementation compatible with official VPoser V02_05."""

from __future__ import annotations

from pathlib import Path


def load_vposer_v02_05(model_dir: str | Path, device="cpu"):
    """Load the official Lightning-format V02_05 checkpoint for pose decoding.

    The PyPI `human_body_prior` release still expects the older `.pt` snapshot
    layout. Official VPoser v2 checkpoints use PyTorch Lightning and prefix
    parameter keys with `vp_model.`; this loader accepts that official layout.
    """
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class BatchFlatten(nn.Module):
        def forward(self, values):
            return values.reshape(values.shape[0], -1)

    class NormalDistDecoder(nn.Module):
        def __init__(self, features: int, latent_dim: int) -> None:
            super().__init__()
            self.mu = nn.Linear(features, latent_dim)
            self.logvar = nn.Linear(features, latent_dim)

        def forward(self, values):
            return torch.distributions.Normal(self.mu(values), functional.softplus(self.logvar(values)))

    class ContinuousRotationDecoder(nn.Module):
        def forward(self, values):
            values = values.view(-1, 3, 2)
            first = functional.normalize(values[:, :, 0], dim=1)
            dot = torch.sum(first * values[:, :, 1], dim=1, keepdim=True)
            second = functional.normalize(values[:, :, 1] - dot * first, dim=1)
            third = torch.cross(first, second, dim=1)
            return torch.stack([first, second, third], dim=-1)

    def matrix_to_axis_angle(rotations):
        """Differentiable SO(3) log map, including the small-angle limit."""
        trace = rotations.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.acos(cosine)
        vee = torch.stack(
            [rotations[:, 2, 1] - rotations[:, 1, 2], rotations[:, 0, 2] - rotations[:, 2, 0], rotations[:, 1, 0] - rotations[:, 0, 1]],
            dim=-1,
        )
        regular = (angle / (2.0 * torch.sin(angle).clamp_min(1e-7))).unsqueeze(-1) * vee
        small = 0.5 * vee
        return torch.where((angle < 1e-4).unsqueeze(-1), small, regular)

    class VPoserV02(nn.Module):
        latentD = 32
        num_joints = 21

        def __init__(self) -> None:
            super().__init__()
            neurons = 512
            self.encoder_net = nn.Sequential(
                BatchFlatten(), nn.BatchNorm1d(63), nn.Linear(63, neurons), nn.LeakyReLU(),
                nn.BatchNorm1d(neurons), nn.Dropout(0.1), nn.Linear(neurons, neurons),
                nn.Linear(neurons, neurons), NormalDistDecoder(neurons, self.latentD),
            )
            self.decoder_net = nn.Sequential(
                nn.Linear(self.latentD, neurons), nn.LeakyReLU(), nn.Dropout(0.1),
                nn.Linear(neurons, neurons), nn.LeakyReLU(), nn.Linear(neurons, self.num_joints * 6),
                ContinuousRotationDecoder(),
            )

        def encode(self, pose_body):
            return self.encoder_net(pose_body)

        def decode(self, latent):
            batch = latent.shape[0]
            rotations = self.decoder_net(latent)
            axis_angle = matrix_to_axis_angle(rotations.reshape(-1, 3, 3)).reshape(batch, self.num_joints, 3)
            return {"pose_body": axis_angle, "pose_body_matrot": rotations.reshape(batch, self.num_joints, 9)}

        def forward(self, pose_body):
            posterior = self.encode(pose_body)
            decoded = self.decode(posterior.rsample())
            decoded.update({"poZ_body_mean": posterior.mean, "poZ_body_std": posterior.scale, "q_z": posterior})
            return decoded

    root = Path(model_dir)
    checkpoints = sorted((root / "snapshots").glob("V02_05*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"no V02_05 checkpoint found under {root / 'snapshots'}")
    checkpoint = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("state_dict", checkpoint)
    state = {name.removeprefix("vp_model."): value for name, value in raw_state.items() if name.startswith("vp_model.")}
    model = VPoserV02()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
