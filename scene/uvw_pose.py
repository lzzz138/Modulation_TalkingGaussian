"""Fixed Gaussian correspondences and sparse, transactional camera refinement."""
import copy
import math
import torch
from data_utils.head_stabilizer.se3 import matrix_multiply
from data_utils.head_stabilizer.se3 import so3_exp


class FixedUVW:
    def __init__(self, uvw, confidence, valid, position, scale):
        self.uvw = uvw.detach().clone()
        self.confidence = confidence.detach().clone().reshape(-1, 1)
        self.valid = valid.detach().clone().bool().reshape(-1, 1)
        self.position = position.detach().clone()
        self.scale = scale.detach().clone().reshape(-1, 1)

    def state_dict(self):
        return {k: getattr(self, k) for k in ('uvw', 'confidence', 'valid', 'position', 'scale')}

    @classmethod
    def restore(cls, state, device):
        return cls(**{k: v.to(device) for k, v in state.items()})

    def select(self, indices):
        for k, value in self.state_dict().items():
            setattr(self, k, value[indices].clone())

    def append_parents(self, parents):
        for k, value in self.state_dict().items():
            setattr(self, k, torch.cat((value, value[parents].clone())))

    def weights(self, xyz):
        distance = (xyz.detach() - self.position).norm(dim=-1, keepdim=True)
        ratio = distance / self.scale.clamp_min(1e-8)
        fade = ((4.0 - ratio) / 2.0).clamp(0, 1)
        return self.confidence * self.valid * fade


def robust_binding(values, weights, yaw_bins, xyz, scales, feature_scale,
                   min_observations=5, residual_limit=0.05):
    """values [frames,points,3], weights [frames,points]."""
    values = torch.nan_to_num(values)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    center = (values * weights[..., None]).sum(0) / weights.sum(0).clamp_min(1e-8)[:, None]
    for _ in range(6):
        residual = ((values - center) / feature_scale).norm(dim=-1)
        robust = weights * (residual_limit / residual.clamp_min(1e-8)).clamp(max=1)
        center = (values * robust[..., None]).sum(0) / robust.sum(0).clamp_min(1e-8)[:, None]
    residual = ((values - center) / feature_scale).norm(dim=-1)
    inliers = (weights > 0) & (residual <= residual_limit)
    coverage = torch.stack([inliers[yaw_bins == b].any(0) for b in yaw_bins.unique()]).sum(0)
    count = inliers.sum(0)
    valid = (count >= min_observations) & (coverage >= 2)
    inlier_weights = weights * inliers
    error = (residual * inlier_weights).sum(0) / inlier_weights.sum(0).clamp_min(1e-8)
    valid &= error <= residual_limit
    confidence = torch.exp(-error / residual_limit) * (count.float() / weights.shape[0])
    return FixedUVW(center, confidence, valid, xyz, scales)


class PoseTable:
    """Each frame has independent Adam state. No unsampled rows can drift."""
    def __init__(self, cameras, device='cuda', lr=1e-3, max_degrees=3.0,
                 max_translation=0.01, acceleration_delay=500,
                 acceleration_ramp=1500):
        self.base = {int(c.talking_dict['img_id']): c.world_view_transform.T.detach().to(device).clone() for c in cameras}
        self.parameters = {i: torch.nn.Parameter(torch.zeros(6, device=device)) for i in self.base}
        self.optimizers = {i: torch.optim.Adam([p], lr=lr) for i, p in self.parameters.items()}
        self.angle = math.radians(max_degrees)
        self.translation_ratio = max_translation
        self.acceleration_delay = int(acceleration_delay)
        self.acceleration_ramp = int(acceleration_ramp)
        self.accepted_updates = 0
        # Dataset world origin is the canonical head center.
        self.center = torch.zeros(3, device=device)

    def matrix(self, camera, differentiable=False):
        i = int(camera.talking_dict['img_id'])
        if i not in self.base:
            return camera.world_view_transform.T.detach()
        base = self.base[i]
        p = self.parameters[i] if differentiable else self.parameters[i].detach()
        rotation = so3_exp(p[:3] * self.angle)
        r = matrix_multiply(rotation, base[:3, :3])
        center_camera = (base[:3, :3] * self.center[None]).sum(-1) + base[:3, 3]
        displacement = p[3:] * center_camera[2].abs().clamp_min(1e-6) * self.translation_ratio
        t = center_camera + displacement - (r * self.center[None]).sum(-1)
        return torch.cat((torch.cat((r, t[:, None]), 1), base[3:4]), 0)

    def regularization(self, i):
        p = self.parameters[i]
        prev = self.parameters.get(i - 1)
        nxt = self.parameters.get(i + 1)
        velocity = p.sum() * 0
        if prev is not None:
            velocity = velocity + (p - prev.detach()).square().mean()
        if nxt is not None:
            velocity = velocity + (nxt.detach() - p).square().mean()
        acceleration = p.sum() * 0
        if prev is not None and nxt is not None:
            acceleration = (nxt.detach() - 2 * p + prev.detach()).square().mean()
        return (0.01 * p.square().mean() + 0.15 * velocity
                + self.acceleration_weight() * acceleration)

    def acceleration_weight(self):
        if self.accepted_updates < self.acceleration_delay:
            return 0.0
        if self.acceleration_ramp <= 0:
            return 0.05
        progress = ((self.accepted_updates - self.acceleration_delay)
                    / float(self.acceleration_ramp))
        return 0.05 * min(max(progress, 0.0), 1.0)

    def update(self, i, objective):
        """Try one transactional update and return JSON-serializable diagnostics."""
        p, optimizer = self.parameters[i], self.optimizers[i]
        before = p.detach().clone()
        state = copy.deepcopy(optimizer.state_dict())
        report = dict(accepted=False, reason=None, old_loss=None,
                      candidate_loss=None, old_coverage=None,
                      candidate_coverage=None, gradient=None,
                      gradient_norm=None, candidate_correction=None,
                      acceleration_weight=self.acceleration_weight(),
                      accepted_updates=self.accepted_updates)
        optimizer.zero_grad(set_to_none=True)
        loss, old_coverage = objective()
        report['old_coverage'] = float(old_coverage.detach() if torch.is_tensor(old_coverage)
                                       else old_coverage)
        if not torch.isfinite(loss):
            report['reason'] = 'old_loss_nonfinite'
            return report
        old_loss = float(loss.detach())
        report['old_loss'] = old_loss
        loss.backward()
        if p.grad is None or not torch.isfinite(p.grad).all():
            report['reason'] = 'gradient_missing_or_nonfinite'
            if p.grad is not None:
                report['gradient'] = p.grad.detach().cpu().tolist()
            optimizer.zero_grad(set_to_none=True)
            return report
        report['gradient'] = p.grad.detach().cpu().tolist()
        report['gradient_norm'] = float(p.grad.detach().norm())
        optimizer.step()
        with torch.no_grad():
            for part in (p[:3], p[3:]):
                part.div_(part.norm().clamp_min(1))
            after_loss, coverage = objective()
            report['candidate_correction'] = p.detach().cpu().tolist()
            report['candidate_coverage'] = float(coverage.detach() if torch.is_tensor(coverage)
                                                 else coverage)
            if torch.isfinite(after_loss):
                report['candidate_loss'] = float(after_loss.detach())
            if not torch.isfinite(after_loss):
                report['reason'] = 'candidate_loss_nonfinite'
            elif coverage < 0.9:
                report['reason'] = 'coverage_below_minimum'
            elif after_loss > old_loss:
                report['reason'] = 'loss_increased'
            else:
                report['accepted'] = True
                report['reason'] = 'accepted'
                self.accepted_updates += 1
                report['accepted_updates'] = self.accepted_updates
            if not report['accepted']:
                p.copy_(before)
                optimizer.load_state_dict(state)
        optimizer.zero_grad(set_to_none=True)
        return report

    def state_dict(self):
        return dict(base=self.base, parameters={i: p.detach() for i, p in self.parameters.items()},
                    optimizers={i: o.state_dict() for i, o in self.optimizers.items()},
                    angle=self.angle, translation_ratio=self.translation_ratio,
                    acceleration_delay=self.acceleration_delay,
                    acceleration_ramp=self.acceleration_ramp,
                    accepted_updates=self.accepted_updates, center=self.center)

    def load_state_dict(self, state):
        if set(state['base']) != set(self.base):
            raise ValueError('Pose frame IDs differ from checkpoint')
        for i, base in self.base.items():
            if not torch.allclose(base, state['base'][i].to(base), atol=1e-6):
                raise ValueError('Initial camera differs from checkpoint')
            self.parameters[i].data.copy_(state['parameters'][i])
            self.optimizers[i].load_state_dict(state['optimizers'][i])
        self.angle = state['angle']
        self.translation_ratio = state['translation_ratio']
        self.acceleration_delay = state.get('acceleration_delay', self.acceleration_delay)
        self.acceleration_ramp = state.get('acceleration_ramp', self.acceleration_ramp)
        self.accepted_updates = state.get('accepted_updates', 0)
        self.center = state['center'].to(self.center)
