"""Differentiable SO(3)/SE(3) helpers shared by preprocessing and training."""

import torch


def matrix_multiply(left, right):
    """Small matrix product without cuBLAS (required by this project's CUDA stack)."""
    return (left.unsqueeze(-1) * right.unsqueeze(-3)).sum(dim=-2)


def matrix_vector(matrix, vector):
    return (matrix * vector.unsqueeze(-2)).sum(dim=-1)


def hat(vector):
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def vee(matrix):
    return torch.stack(
        (matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]), dim=-1
    )


def _series_coefficients(theta_sq):
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-16))
    small = theta_sq < 1e-8
    a = torch.where(small, 1.0 - theta_sq / 6.0 + theta_sq.square() / 120.0,
                    torch.sin(theta) / torch.clamp(theta, min=1e-8))
    b = torch.where(small, 0.5 - theta_sq / 24.0 + theta_sq.square() / 720.0,
                    (1.0 - torch.cos(theta)) / torch.clamp(theta_sq, min=1e-16))
    c = torch.where(small, 1.0 / 6.0 - theta_sq / 120.0 + theta_sq.square() / 5040.0,
                    (theta - torch.sin(theta)) / torch.clamp(theta_sq * theta, min=1e-24))
    return a, b, c


def so3_exp(omega):
    theta_sq = omega.square().sum(dim=-1, keepdim=True)
    a, b, _ = _series_coefficients(theta_sq)
    omega_hat = hat(omega)
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(
        omega.shape[:-1] + (3, 3)
    )
    return identity + a[..., None] * omega_hat + b[..., None] * matrix_multiply(omega_hat, omega_hat)


def so3_log(rotation):
    cos_theta = (rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    skew = 0.5 * (rotation - rotation.transpose(-1, -2))
    scale = torch.where(theta < 1e-4, 1.0 + theta.square() / 6.0,
                        theta / torch.clamp(torch.sin(theta), min=1e-7))
    return scale[..., None] * vee(skew)


def se3_exp(twist):
    omega, velocity = twist[..., :3], twist[..., 3:]
    theta_sq = omega.square().sum(dim=-1, keepdim=True)
    _, b, c = _series_coefficients(theta_sq)
    omega_hat = hat(omega)
    identity = torch.eye(3, dtype=twist.dtype, device=twist.device).expand(
        twist.shape[:-1] + (3, 3)
    )
    jacobian = identity + b[..., None] * omega_hat + c[..., None] * matrix_multiply(omega_hat, omega_hat)
    return so3_exp(omega), matrix_vector(jacobian, velocity)


def se3_log(rotation, translation):
    omega = so3_log(rotation)
    theta_sq = omega.square().sum(dim=-1, keepdim=True)
    omega_hat = hat(omega)
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand(
        rotation.shape[:-2] + (3, 3)
    )
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-16))
    half = 0.5 * theta
    coefficient = torch.where(
        theta_sq < 1e-8,
        1.0 / 12.0 + theta_sq / 720.0,
        (1.0 - half * torch.cos(half) / torch.clamp(torch.sin(half), min=1e-7))
        / torch.clamp(theta_sq, min=1e-16),
    )
    jacobian_inv = identity - 0.5 * omega_hat + coefficient[..., None] * matrix_multiply(omega_hat, omega_hat)
    velocity = matrix_vector(jacobian_inv, translation)
    return torch.cat((omega, velocity), dim=-1)


def compose_increment(twist, base_rotation, base_translation):
    delta_rotation, delta_translation = se3_exp(twist)
    rotation = matrix_multiply(delta_rotation, base_rotation)
    translation = matrix_vector(delta_rotation, base_translation) + delta_translation
    return rotation, translation
