"""Geometric angles with finite gradients at parallel/antiparallel vectors."""
import torch


def angle_between(first, second):
    dot = (first * second).sum(dim=-1)
    cross = torch.linalg.vector_norm(torch.cross(first, second, dim=-1), dim=-1)
    degenerate = (torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(second, dim=-1)) < 1e-8
    return torch.atan2(torch.where(degenerate, torch.zeros_like(cross), cross),
                       torch.where(degenerate, torch.ones_like(dot), dot))


def flexion(first, centre, last):
    return torch.pi - angle_between(first-centre, last-centre)


def inclination(vector):
    vertical = torch.zeros_like(vector)
    vertical[..., 1] = 1
    return angle_between(vector, vertical)
