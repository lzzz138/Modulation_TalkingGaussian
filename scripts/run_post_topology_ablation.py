#!/usr/bin/env python3
"""Run the five-video post-topology D group on GPUs 2/3 and compare A/B/C/D."""

import argparse
import csv
import datetime
import os
import subprocess
import time


VIDEOS = ('posevideo5', 'posevideo', 'posevideo2', 'posevideo3', 'posevideo4')
GROUPS = (('A', 'coarse_qgsfm'), ('B', 'cfhs_qgsfm'),
          ('C', 'cfhs_uvw_qgsfm'), ('D', 'post_topology_uvw_qgsfm'))
METRICS = ('PSNR', 'SSIM', 'LPIPS', 'tLPIPS', 'FID', 'LMD_px', 'Stability',
           'LVD_px', 'Flow_Warp_Error_excess', 'tOF_px')
REQUIRED_DATA = ('transforms_train.json', 'transforms_val.json',
                 'track_params_canonical.pt', 'canonical_features/manifest.json',
                 'canonical_diagnostics.json', 'aud.npy', 'au.csv')


def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def available_gpus():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
         '--format=csv,noheader,nounits'], universal_newlines=True)
    states, uuids = {}, {}
    for line in output.splitlines():
        index, uuid, free, utilization = (part.strip() for part in line.split(','))
        states[int(index)] = (int(free), int(utilization))
        uuids[uuid] = int(index)
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
         '--format=csv,noheader,nounits'], universal_newlines=True)
    busy = {uuids[line.split(',', 1)[0].strip()] for line in output.splitlines()
            if line.strip() and line.split(',', 1)[0].strip() in uuids}
    return states, busy


def read_metrics(path):
    values = {}
    with open(path) as stream:
        for line in stream:
            key, separator, value = line.partition(':')
            if separator and key in METRICS:
                try:
                    values[key] = float(value.strip().split()[0])
                except (IndexError, ValueError):
                    pass
    missing = set(METRICS) - set(values)
    if missing:
        raise ValueError('%s is missing metrics %s' % (path, sorted(missing)))
    return values


def write_comparison(root):
    rows = []
    for video in VIDEOS:
        for group, directory in GROUPS:
            output = os.path.join(root, video, directory)
            if group == 'D' and not os.path.isfile(os.path.join(output, '.metrics_complete')):
                continue
            path = os.path.join(output, 'metrics_full.txt')
            if os.path.isfile(path):
                rows.append(dict(video=video, group=group, **read_metrics(path)))
    if not rows:
        return
    csv_path = os.path.join(root, 'post_topology_ablation_summary.csv')
    with open(csv_path, 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=('video', 'group') + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    complete = [video for video in VIDEOS if all(
        any(row['video'] == video and row['group'] == group for row in rows)
        for group, _ in GROUPS)]
    markdown_path = os.path.join(root, 'post_topology_ablation_summary.md')
    with open(markdown_path, 'w') as stream:
        stream.write('# A/B/C/D video ablation\n\n')
        stream.write('A: coarse QGSFM; B: CFHS QGSFM; C: original joint UVW; '
                     'D: UVW bound after Face topology freezes at 40k.\n\n')
        stream.write('| Video | Group | ' + ' | '.join(METRICS) + ' |\n')
        stream.write('| --- | --- | ' + ' | '.join('---:' for _ in METRICS) + ' |\n')
        for row in rows:
            stream.write('| %s | %s | %s |\n' % (
                row['video'], row['group'],
                ' | '.join('%.6f' % row[key] for key in METRICS)))
        if complete:
            stream.write('\n## Mean over videos with all four groups\n\n')
            stream.write('| Group | ' + ' | '.join(METRICS) + ' |\n')
            stream.write('| --- | ' + ' | '.join('---:' for _ in METRICS) + ' |\n')
            for group, _ in GROUPS:
                subset = [row for row in rows if row['video'] in complete and row['group'] == group]
                stream.write('| %s | %s |\n' % (
                    group, ' | '.join('%.6f' % (sum(row[key] for row in subset) / len(subset))
                                      for key in METRICS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='2,3')
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--minimum-free-mib', type=int, default=12000)
    parser.add_argument('--maximum-utilization', type=int, default=25)
    parser.add_argument('--repo', default='/home/lzq/paperCode/talkinghead/TalkingGaussian')
    parser.add_argument('--data-root', default='/mnt/folder_2/lzq/TalkingGaussian/ablation_data')
    parser.add_argument('--output-root', default='/mnt/folder_2/lzq/TalkingGaussian/ablation_output')
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(','))
    if not gpus or not set(gpus).issubset({2, 3}):
        parser.error('D-group jobs are restricted to GPUs 2 and 3')
    if args.poll_seconds < 1:
        parser.error('poll-seconds must be positive')
    conda_bin = '/home/lzq/miniconda3/envs/talking_gaussian/bin'
    env = os.environ.copy()
    env['PATH'] = conda_bin + os.pathsep + env.get('PATH', '')
    env['CONDA_PREFIX'] = os.path.dirname(conda_bin)
    env['CONDA_DEFAULT_ENV'] = 'talking_gaussian'
    env['PYTHONUNBUFFERED'] = '1'
    pending = []
    for video in VIDEOS:
        dataset = os.path.join(args.data_root, video, 'cfhs_uvw_qgsfm')
        for name in REQUIRED_DATA:
            if not os.path.isfile(os.path.join(dataset, name)):
                raise FileNotFoundError('%s is missing %s' % (dataset, name))
        output = os.path.join(args.output_root, video, GROUPS[-1][1])
        if not os.path.isfile(os.path.join(output, '.metrics_complete')):
            pending.append(video)

    running, failures = {}, []
    while pending or running:
        for gpu, job in list(running.items()):
            process, log, video, output, phase = job
            code = process.poll()
            if code is None:
                continue
            log.close()
            del running[gpu]
            if code:
                with open(os.path.join(output, '.ablation_failed'), 'w') as stream:
                    stream.write('%s phase=%s exit=%d\n' % (timestamp(), phase, code))
                failures.append('%s %s exit=%d' % (video, phase, code))
                print('[%s] failed %s %s exit=%d' % (timestamp(), video, phase, code), flush=True)
                continue
            if phase == 'train':
                with open(os.path.join(output, '.ablation_complete'), 'w') as stream:
                    stream.write('%s gpu=%d\n' % (timestamp(), gpu))
                process, log = start_metrics(conda_bin, args.repo, output, gpu, env)
                running[gpu] = (process, log, video, output, 'metrics')
                print('[%s] metrics %s GPU %d pid=%d' %
                      (timestamp(), video, gpu, process.pid), flush=True)
            else:
                try:
                    read_metrics(os.path.join(output, 'metrics_full.txt'))
                except ValueError as error:
                    failures.append(str(error))
                    print('[%s] failed metrics %s: %s' % (timestamp(), video, error), flush=True)
                    continue
                with open(os.path.join(output, '.metrics_complete'), 'w') as stream:
                    stream.write('%s\n' % timestamp())
                write_comparison(args.output_root)
                print('[%s] complete %s' % (timestamp(), video), flush=True)
        if pending:
            states, busy = available_gpus()
            for gpu in gpus:
                if gpu in running or gpu in busy or not pending:
                    continue
                free, utilization = states[gpu]
                if free < args.minimum_free_mib or utilization > args.maximum_utilization:
                    continue
                video = pending.pop(0)
                output = os.path.join(args.output_root, video, GROUPS[-1][1])
                os.makedirs(output, exist_ok=True)
                if os.path.isfile(os.path.join(output, '.ablation_complete')):
                    process, log = start_metrics(conda_bin, args.repo, output, gpu, env)
                    phase = 'metrics'
                else:
                    dataset = os.path.join(args.data_root, video, 'cfhs_uvw_qgsfm')
                    checkpoint = os.path.join(output, 'chkpnt_uvw_post_topology_latest.pth')
                    command = ['bash', 'scripts/train_uvw_pose_post_topology.sh',
                               dataset, output, str(gpu), '1']
                    if os.path.isfile(checkpoint):
                        command.append(checkpoint)
                    log = open(os.path.join(output, 'train.log'), 'a')
                    log.write('[%s] %s\n' % (timestamp(), ' '.join(command)))
                    log.flush()
                    process = subprocess.Popen(command, cwd=args.repo, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                    phase = 'train'
                running[gpu] = (process, log, video, output, phase)
                print('[%s] started %s %s GPU %d pid=%d' %
                      (timestamp(), video, phase, gpu, process.pid), flush=True)
        if pending or running:
            time.sleep(args.poll_seconds)
    write_comparison(args.output_root)
    if failures:
        raise RuntimeError('D-group failures: ' + '; '.join(failures))
    print('[%s] all five D-group videos complete' % timestamp(), flush=True)


def start_metrics(conda_bin, repo, output, gpu, env):
    log = open(os.path.join(output, 'metrics_full.txt'), 'w')
    gpu_env = env.copy()
    gpu_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    command = [os.path.join(conda_bin, 'python'), '-u', 'metric.py',
               os.path.join(output, 'test/ours_None/renders/out.mp4'),
               os.path.join(output, 'test/ours_None/gt/out.mp4')]
    return subprocess.Popen(command, cwd=repo, env=gpu_env,
                            stdout=log, stderr=subprocess.STDOUT), log


if __name__ == '__main__':
    main()
