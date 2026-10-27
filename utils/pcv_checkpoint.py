"""Completed-step PCV checkpoints, including camera order and sampling RNG."""
import ast
import hashlib
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch

CONFIG_KEYS = ('pose_visibility_deformation', 'pcv_start', 'pcv_ramp_steps',
               'pcv_max_displacement', 'pcv_visibility_resolution', 'pcv_lr',
               'pcv_reg_weight', 'pcv_condition', 'geometry_mod_multiscale',
               'geometry_mod_map_res', 'geometry_mod_condition_scale', 'audio_extractor', 'sh_degree')

def config(args):
    return {k: getattr(args, k) for k in CONFIG_KEYS}

def sha256(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def data_identity(path):
    root = Path(path)
    audio = root/'aud_ds.npy'
    if not audio.exists(): audio = root/'aud.npy'
    return dict(transforms_train=sha256(root/'transforms_train.json'),
                transforms_val=sha256(root/'transforms_val.json'),
                au=sha256(root/'au.csv'), audio=sha256(audio))


def random_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())

def restore_random(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state[3]])

def sampling(scene, stack):
    ids = lambda cameras: [int(c.talking_dict['img_id']) for c in cameras]
    return dict(order=ids(scene.getTrainCameras()), stack=ids(stack or []), rng=random_state())

def restore_sampling(scene, state):
    cameras = {int(c.talking_dict['img_id']): c for c in scene.getTrainCameras()}
    if set(cameras) != set(state['order']) or len(cameras) != len(state['order']):
        raise ValueError('PCV checkpoint training frame IDs differ')
    scene.getTrainCameras()[:] = [cameras[i] for i in state['order']]
    restore_random(state['rng'])
    return [cameras[i] for i in state['stack']]

def atomic_save(state, path):
    temporary = str(path)+'.tmp'
    torch.save(state, temporary); os.replace(temporary, str(path))

def save_face(path, args, opt, scene, gaussian, motion, optimizer, scheduler, iteration, stack, identity):
    state = dict(format='pcv_face_v1', config=config(args), optimization=vars(opt).copy(),
                 identity=identity, iteration=iteration, gaussians=gaussian.capture(),
                 motion=motion.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 sampling=sampling(scene, stack))
    atomic_save(state, path)
    return state

def restore_face(state, args, opt, scene, gaussian, motion, optimizer, scheduler, identity):
    if not isinstance(state, dict) or state.get('format') != 'pcv_face_v1':
        raise ValueError('PCV training requires a complete PCV checkpoint')
    expected, saved = config(args), state['config']
    if getattr(args, 'pcv_fork_6k', False):
        if state['iteration'] != 6000 or saved['pcv_start'] != 6000 or float(state['motion']['pose_residual.strength']) != 0:
            raise ValueError('Ablations can fork only the completed 6000-step zero-strength checkpoint')
        allowed = {'pcv_condition','pcv_max_displacement','pcv_reg_weight'}
        if any(saved[k] != expected[k] for k in expected if k not in allowed):
            raise ValueError('PCV fork changed an unrelated setting')
    elif saved != expected:
        raise ValueError('PCV training configuration mismatch')
    if state['identity'] != identity or state['optimization'] != vars(opt):
        raise ValueError('PCV dataset or optimization settings mismatch')
    gaussian.restore(state['gaussians'], opt); motion.load_state_dict(state['motion'], strict=True)
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    return state['iteration'], restore_sampling(scene, state['sampling'])

def apply_saved_config(state, args):
    if state.get('format') not in ('pcv_face_v1','pcv_fuse_v1'):
        raise ValueError('Unsupported PCV checkpoint format')
    if getattr(args, 'uvw_pose', False): raise ValueError('PCV cannot load online poses')
    for key, value in state['config'].items():
        explicit = any(a == '--'+key or a.startswith('--'+key+'=') for a in sys.argv[1:])
        if explicit and getattr(args, key) != value:
            raise ValueError('Explicit option differs from checkpoint: '+key)
        setattr(args, key, value)

def validate_mouth(path, source, device="cpu"):
    # Legacy Mouth checkpoints carry no identity; require their saved source config.
    path = Path(path)
    proof = path.parent/'pcv_mouth_source.json'
    if proof.exists():
        metadata = json.loads(proof.read_text())
        if metadata['checkpoint_sha256'] != sha256(path) or metadata['identity'] != data_identity(source):
            raise ValueError('Mouth checkpoint provenance mismatch')
    else:
        cfg = ast.parse((path.parent/'cfg_args').read_text(), mode='eval').body
        values = {k.arg: ast.literal_eval(k.value) for k in cfg.keywords}
        if values.get('uvw_pose') or data_identity(values['source_path']) != data_identity(source):
            raise ValueError('Mouth checkpoint must come from the same fixed-camera B dataset')
    state = torch.load(str(path), map_location=device)
    if not isinstance(state, tuple) or len(state) != 4 or state[-1] != 50000:
        raise ValueError('Expected an unfused original Mouth 50k checkpoint')
    return state

if __name__ == '__main__':
    import argparse, shutil
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouth', required=True); parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    validate_mouth(a.mouth, a.source)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    target = out/'chkpnt_mouth_latest.pth'
    if target.exists(): raise ValueError('Refusing to overwrite an existing Mouth checkpoint')
    shutil.copy2(a.mouth, str(target))
    (out/'pcv_mouth_source.json').write_text(json.dumps(dict(identity=data_identity(a.source),
        checkpoint_sha256=sha256(target), source_checkpoint=str(Path(a.mouth).resolve())), indent=2))
