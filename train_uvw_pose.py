"""Alternating UVW camera and independent Face/Mouth reconstruction updates."""
import copy
import json
import os
import random
from argparse import ArgumentParser
import numpy as np
import torch
from data_utils.head_stabilizer.se3 import matrix_multiply
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from scene.uvw_pose import PoseTable, FixedUVW, robust_binding
from scene.uvw_observations import preflight, Observations, sample
from gaussian_renderer.uvw import render_uvw
from utils.camera_utils import loadCamOnTheFly
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import safe_state


def corrected_camera(camera, table):
    result = copy.copy(camera)
    w = table.matrix(camera).detach()
    result.world_view_transform = w.T.contiguous()
    result.full_proj_transform = matrix_multiply(result.world_view_transform, result.projection_matrix)
    result.camera_center = -((w[:3, :3].T * w[:3, 3][None]).sum(-1))
    return result


class Runtime:
    def __init__(self, dataset, options, pipe, args):
        self.dataset, self.options, self.pipe, self.args = dataset, options, pipe, args
        self.params, self.cache, self.digest = preflight(dataset.source_path)
        self.resume = torch.load(args.resume, map_location='cuda') if args.resume else None
        if self.resume and (self.resume.get('format') != 'uvw_pose_v1' or self.resume['dataset_digest'] != self.digest):
            raise ValueError('Incompatible UVW checkpoint or dataset')
        if self.resume and self.resume['iteration'] >= 3000 and self.resume['branches']['face']['uvw'] is None:
            raise ValueError('Post-warmup UVW checkpoint is missing fixed bindings')
        if self.resume and self.resume['config'] != self.configuration():
            raise ValueError('UVW resume training configuration differs')
        self.scene = None
        self.table = None
        self.binding_ids = []
        self.feature_scale = None
        self.log = None

    def configuration(self):
        return dict(iterations=self.options.iterations, interval=self.args.pose_interval,
                    stop=self.args.pose_stop, pose_enabled=not self.args.fixed_pose_control,
                    model={k: v for k, v in vars(self.dataset).items() if k not in ('model_path', 'source_path', 'eval')},
                    optimization=vars(self.options).copy(),
                    binding=dict(max_frames=64, yaw_bin_degrees=15, min_observations=5, min_yaw_bins=2, residual_limit=0.05),
                    pose=dict(lr=1e-3, max_degrees=3.0,
                              translation_ratio=0.01, rgb=0.05, prior=0.01,
                              velocity=0.15, acceleration=0.05,
                              acceleration_delay=self.args.pose_acceleration_delay,
                              acceleration_ramp=self.args.pose_acceleration_ramp,
                              coverage=0.1, min_pixels=512, retention=0.9))

    def make_scene(self, dataset, gaussian, branch):
        if self.scene is None:
            self.scene = Scene(dataset, gaussian)
            self.source_cameras = list(self.scene.getTrainCameras())
            expected_ids = {int(f["img_id"]) for f in json.load(open(os.path.join(dataset.source_path, "transforms_train.json")))["frames"]}
            if {int(c.talking_dict["img_id"]) for c in self.source_cameras} != expected_ids:
                raise ValueError("Online pose optimization requires training cameras only")
            self.table = PoseTable(
                self.source_cameras,
                acceleration_delay=self.args.pose_acceleration_delay,
                acceleration_ramp=self.args.pose_acceleration_ramp,
            )
            self.by_id = {int(c.talking_dict['img_id']): c for c in self.source_cameras}
            self.observations = Observations(dataset.source_path, self.source_cameras,
                                             self.params, self.cache, dataset.model_path)
            if self.resume:
                self.table.load_state_dict(self.resume['pose'])
                self.binding_ids = self.resume['binding_ids']
                self.feature_scale = self.resume['feature_scale']
        else:
            xyz = np.random.random((dataset.init_num, 3)) * 0.2 - 0.1
            gaussian.create_from_pcd(BasicPointCloud(points=xyz, colors=np.random.random(xyz.shape) / 255 + 0.5,
                                                    normals=np.zeros_like(xyz)), self.scene.cameras_extent)
        scene = copy.copy(self.scene)
        scene.gaussians = gaussian
        scene.getTrainCameras = lambda scale=1.0: [self.camera(c) for c in self.source_cameras]
        return scene

    def camera(self, camera):
        return corrected_camera(camera, self.table)

    def restore_branch(self, branch, gaussian, motion, optimizer, scheduler):
        if not self.resume:
            return 0, None
        state = self.resume['branches'][branch]
        gaussian.restore(state['gaussians'], self.options)
        motion.load_state_dict(state['motion'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if state['uvw'] is not None:
            gaussian.fixed_uvw = FixedUVW.restore(state['uvw'], 'cuda')
        for key in state.get('frozen_gaussians', []):
            getattr(gaussian, key).requires_grad_(False)
        for name, p in motion.named_parameters():
            if name in state.get('frozen_motion', []):
                p.requires_grad_(False)
        return self.resume['iteration'], [self.by_id[i] for i in state['stack']]

    def loaded_camera(self, i):
        c = self.by_id[i]
        return c if c.original_image is not None else loadCamOnTheFly(copy.deepcopy(c))

    @torch.no_grad()
    def bind(self, face):
        gaussian = face['gaussians']
        self.binding_ids = self.observations.selected()
        values, weights, bins = [], [], []
        bg = torch.zeros(3, device='cuda')
        for i in self.binding_ids:
            c = self.loaded_camera(i)
            obs = self.observations.get(i)
            out = render_uvw(c, gaussian, None, self.pipe, bg, freeze_field=True, render_attributes=False)
            w = c.world_view_transform.T
            xyz = matrix_multiply(gaussian.get_xyz, w[:3, :3].T) + w[:3, 3]
            clip = matrix_multiply(torch.cat((xyz, torch.ones_like(xyz[:, :1])), -1), c.projection_matrix)
            ndc = clip[:, :2] / clip[:, 3:].clamp_min(1e-8)
            xy = (ndc + 1) * xyz.new_tensor([c.image_width, c.image_height]) / 2 - 0.5
            a = sample(out['alpha'], xy, c.image_width, c.image_height)[:, 0]
            depth = sample(out['depth'], xy, c.image_width, c.image_height)[:, 0] / a.clamp_min(1e-6)
            mask = sample(obs['mask'][None].float(), xy, c.image_width, c.image_height)[:, 0]
            visible = (a > 0.5) & (mask > 0.99) & (xyz[:, 2] > 0)
            visible &= (depth - xyz[:, 2]).abs() < torch.maximum(gaussian.get_scaling.max(-1).values * 2, depth.abs() * 0.01)
            uvw = sample(obs['uvw'], xy, c.image_width, c.image_height)
            visible &= torch.isfinite(uvw).all(-1)
            values.append(uvw)
            weights.append(visible.float() * self.observations.conf.get(i, 0))
            bins.append(round(self.observations.yaw[i] / 15))
        values, weights = torch.stack(values), torch.stack(weights)
        valid_values = values[weights > 0]
        self.feature_scale = (torch.quantile(valid_values, 0.95, dim=0) - torch.quantile(valid_values, 0.05, dim=0)).clamp_min(1e-3) if len(valid_values) else torch.ones(3, device='cuda')
        gaussian.fixed_uvw = robust_binding(values, weights, torch.tensor(bins, device='cuda'),
                                           gaussian.get_xyz, gaussian.get_scaling.max(-1).values,
                                           self.feature_scale)
        report = dict(frame_ids=self.binding_ids, yaw_bins=bins, reliable_gaussians=int(gaussian.fixed_uvw.valid.sum()), total_gaussians=len(gaussian.get_xyz), feature_scale=self.feature_scale.cpu().tolist(), config=self.configuration()['binding'])
        with open(os.path.join(self.dataset.model_path, 'uvw_binding.json'), 'w') as stream:
            json.dump(report, stream, indent=2)
        print('Fixed UVW: %d/%d reliable Gaussians' % (gaussian.fixed_uvw.valid.sum(), len(gaussian.get_xyz)))

    def pose_step(self, face, iteration):
        gaussian, motion = face['gaussians'], face['motion']
        if gaussian.fixed_uvw is None or not gaussian.fixed_uvw.valid.any():
            return dict(status='no_bindings', iteration=iteration)
        i = random.choice(sorted(self.by_id))
        c = self.loaded_camera(i)
        obs = self.observations.get(i)
        bg = torch.zeros(3, device='cuda')
        with torch.no_grad():
            baseline = render_uvw(c, gaussian, motion, self.pipe, bg, self.table.matrix(c), True)
            mask = obs['mask'] & (baseline['uvw_coverage'][0] > 0.1) & (baseline['alpha'][0] > 0.5)
        count = int(mask.sum())
        if count < max(512, int(obs['mask'].sum() * 0.1)):
            return dict(status='insufficient_coverage', iteration=iteration, frame=i, pixels=count)
        target = c.original_image.cuda().float() / 255

        def objective():
            out = render_uvw(c, gaussian, motion, self.pipe, bg, self.table.matrix(c, True), True)
            residual = (out['uvw'] - obs['uvw']) / self.feature_scale[:, None, None]
            uvw_loss = torch.sqrt(residual[:, mask].square() + 1e-6).mean()
            rgb_loss = (out['render'][:, mask] - target[:, mask]).abs().mean()
            coverage = ((out['uvw_coverage'][0] > 0.1) & (out['alpha'][0] > 0.5))[mask].float().mean()
            return uvw_loss + 0.05 * rgb_loss + self.table.regularization(i), coverage

        attempt = self.table.update(i, objective)
        attempt.update(
            status='accepted' if attempt['accepted'] else 'rejected',
            iteration=iteration,
            frame=i,
            pixels=count,
            correction=self.table.parameters[i].detach().cpu().tolist(),
        )
        return attempt

    def save(self, states, iteration):
        branches = {}
        for name, s in states.items():
            g = s['gaussians']
            branches[name] = dict(gaussians=g.capture(), motion=s['motion'].state_dict(),
                                  optimizer=s['optimizer'].state_dict(), scheduler=s['scheduler'].state_dict(),
                                  uvw=None if g.fixed_uvw is None else g.fixed_uvw.state_dict(),
                                  stack=[int(c.talking_dict['img_id']) for c in (s['stack'] or [])],
                                  frozen_gaussians=[k for k in ('_xyz','_opacity','_scaling','_rotation') if not getattr(g,k).requires_grad],
                                  frozen_motion=[n for n,p in s['motion'].named_parameters() if not p.requires_grad])
        state = dict(format='uvw_pose_v1', iteration=iteration, dataset_digest=self.digest,
                     branches=branches, pose=self.table.state_dict(), binding_ids=self.binding_ids,
                     feature_scale=self.feature_scale, config=self.configuration(),
                     rng=(random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()))
        path = os.path.join(self.dataset.model_path, 'chkpnt_uvw_latest.pth')
        torch.save(state, path + '.tmp')
        os.replace(path + '.tmp', path)
        self.export_transforms()

    def export_transforms(self):
        document = json.load(open(os.path.join(self.dataset.source_path, 'transforms_train.json')))
        for f in document['frames']:
            camera = self.by_id[int(f['img_id'])]
            w = self.table.matrix(camera).detach()
            c2w = torch.linalg.inv(w).cpu()
            c2w[:3, 1:3] *= -1
            f['transform_matrix'] = c2w.tolist()
        with open(os.path.join(self.dataset.model_path, 'transforms_train_refined.json'), 'w') as out:
            json.dump(document, out, indent=2)


def main():
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('--resume')
    parser.add_argument('--stop_after', type=int, help='Stop early without changing the training schedule')
    parser.add_argument('--pose_acceleration_delay', type=int, default=500,
                        help='Accepted pose updates before enabling acceleration loss')
    parser.add_argument('--pose_acceleration_ramp', type=int, default=1500,
                        help='Accepted updates used to ramp acceleration weight to 0.05')
    parser.add_argument('--pose_interval', type=int, default=5)
    parser.add_argument('--pose_stop', type=int, default=45000)
    parser.add_argument('--fixed_pose_control', action='store_true')
    parser.add_argument('--checkpoint_every', type=int, default=1000)
    args = parser.parse_args()
    if (args.pose_interval < 1 or args.checkpoint_every < 1
            or args.pose_acceleration_delay < 0
            or args.pose_acceleration_ramp < 0):
        parser.error('intervals must be positive')
    args.eval = False
    args.uvw_pose = True
    safe_state(False)
    dataset, options, pipe = model.extract(args), optimization.extract(args), pipeline.extract(args)
    os.makedirs(dataset.model_path, exist_ok=True)
    runtime = Runtime(dataset, options, pipe, args)
    import train_face
    import train_mouth
    generators, states = {}, {}
    for branch, module, count in (('face', train_face, 2000), ('mouth', train_mouth, 10000)):
        d, o = copy.copy(dataset), copy.copy(options)
        d.init_num = count
        if branch == 'face':
            o.densify_grad_threshold = 0.0005
        generators[branch] = module.training_steps(d, o, pipe, [], [], [], None, -1, runtime)
        states[branch] = next(generators[branch])
    if runtime.resume:
        py, np_state, cpu, cuda = runtime.resume['rng']
        random.setstate(py); np.random.set_state(np_state)
        torch.set_rng_state(cpu.cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
    start = states['face']['iteration']
    with open(os.path.join(dataset.model_path, 'pose_updates.jsonl'), 'a') as log:
        last = min(options.iterations, args.stop_after or options.iterations)
        for iteration in range(start + 1, last + 1):
            for branch in generators:
                states[branch] = next(generators[branch])
                assert states[branch]['iteration'] == iteration
            if iteration == 3000:
                runtime.bind(states['face'])
            if 3000 < iteration < args.pose_stop and iteration % args.pose_interval == 0 and not args.fixed_pose_control:
                report = runtime.pose_step(states['face'], iteration)
                log.write(json.dumps(report) + '\n'); log.flush()
            if iteration % args.checkpoint_every == 0 or iteration == last:
                runtime.save(states, iteration)


if __name__ == '__main__':
    main()
