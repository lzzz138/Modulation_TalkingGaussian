import math

import torch


def hat(vector):
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def vee(matrix):
    return torch.stack(
        (matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]), dim=-1
    )


def _series_coefficients(theta_sq):
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-16))
    small = theta_sq < 1e-8
    a = torch.where(
        small,
        1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0,
        torch.sin(theta) / torch.clamp(theta, min=1e-8),
    )
    b = torch.where(
        small,
        0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0,
        (1.0 - torch.cos(theta)) / torch.clamp(theta_sq, min=1e-16),
    )
    c = torch.where(
        small,
        1.0 / 6.0 - theta_sq / 120.0 + theta_sq * theta_sq / 5040.0,
        (theta - torch.sin(theta))
        / torch.clamp(theta_sq * theta, min=1e-24),
    )
    return a, b, c


def so3_exp(omega):
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)
    a, b, _ = _series_coefficients(theta_sq)
    omega_hat = hat(omega)
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device)
    identity = identity.expand(omega.shape[:-1] + (3, 3))
    return identity + a[..., None] * omega_hat + b[..., None] * torch.matmul(
        omega_hat, omega_hat
    )


def so3_log(rotation):
    cos_theta = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    skew = 0.5 * (rotation - rotation.transpose(-1, -2))
    sin_theta = torch.sin(theta)
    scale = torch.where(
        theta < 1e-4,
        1.0 + theta * theta / 6.0,
        theta / torch.clamp(sin_theta, min=1e-7),
    )
    return scale[..., None] * vee(skew)


def se3_exp(twist):
    omega = twist[..., :3]
    velocity = twist[..., 3:]
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)
    _, b, c = _series_coefficients(theta_sq)
    omega_hat = hat(omega)
    identity = torch.eye(3, dtype=twist.dtype, device=twist.device)
    identity = identity.expand(twist.shape[:-1] + (3, 3))
    jacobian = identity + b[..., None] * omega_hat + c[..., None] * torch.matmul(
        omega_hat, omega_hat
    )
    translation = torch.matmul(jacobian, velocity[..., None]).squeeze(-1)
    return so3_exp(omega), translation


def se3_log(rotation, translation):
    omega = so3_log(rotation)
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)
    omega_hat = hat(omega)
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    identity = identity.expand(rotation.shape[:-2] + (3, 3))
    theta = torch.sqrt(torch.clamp(theta_sq, min=1e-16))
    half = 0.5 * theta
    coefficient = torch.where(
        theta_sq < 1e-8,
        1.0 / 12.0 + theta_sq / 720.0,
        (1.0 - half * torch.cos(half) / torch.clamp(torch.sin(half), min=1e-7))
        / torch.clamp(theta_sq, min=1e-16),
    )
    jacobian_inv = identity - 0.5 * omega_hat + coefficient[..., None] * torch.matmul(
        omega_hat, omega_hat
    )
    velocity = torch.matmul(jacobian_inv, translation[..., None]).squeeze(-1)
    return torch.cat((omega, velocity), dim=-1)


def compose_increment(twist, base_rotation, base_translation):
    delta_rotation, delta_translation = se3_exp(twist)
    rotation = torch.matmul(delta_rotation, base_rotation)
    translation = torch.matmul(
        delta_rotation, base_translation[..., None]
    ).squeeze(-1) + delta_translation
    return rotation, translation


def relative_twist(rotation, translation):
    previous_inv = rotation[:-1].transpose(-1, -2)
    relative_rotation = torch.matmul(rotation[1:], previous_inv)
    relative_translation = translation[1:] - torch.matmul(
        relative_rotation, translation[:-1, :, None]
    ).squeeze(-1)
    return se3_log(relative_rotation, relative_translation)


def rotation_to_euler(rotation):
    # Inverse of the repository convention R = Rx * Ry * Rz.
    sy = torch.clamp(rotation[..., 0, 2], -1.0, 1.0)
    y = torch.asin(sy)
    cy = torch.cos(y)
    regular = torch.abs(cy) > 1e-6
    x_regular = torch.atan2(-rotation[..., 1, 2], rotation[..., 2, 2])
    z_regular = torch.atan2(rotation[..., 0, 1], rotation[..., 0, 0])
    x_gimbal = torch.atan2(rotation[..., 2, 1], rotation[..., 1, 1])
    x = torch.where(regular, x_regular, x_gimbal)
    z = torch.where(regular, z_regular, torch.zeros_like(z_regular))
    return torch.stack((x, y, z), dim=-1)


def adaptive_temporal_weight(base_twist, depth):
    scaled = base_twist.clone()
    scaled[..., 3:] = scaled[..., 3:] / depth
    motion = torch.linalg.norm(scaled, dim=-1)
    half_motion = math.radians(3.0)
    return torch.clamp(torch.exp(-math.log(2.0) * motion / half_motion), 0.1, 1.0)
