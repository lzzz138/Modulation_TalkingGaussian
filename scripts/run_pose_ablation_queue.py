#!/usr/bin/env python3
"""Run the five-video A/B/C pose ablation when selected GPUs become idle."""

import argparse
import datetime
import os
import subprocess
import time


VIDEOS = ("posevideo", "posevideo2", "posevideo3", "posevideo4", "posevideo5")
GROUPS = ("coarse_qgsfm", "cfhs_qgsfm", "cfhs_uvw_qgsfm")


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def gpu_state():
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    states = {}
    uuid_to_index = {}
    for line in query.splitlines():
        index, uuid, free, utilization = [item.strip() for item in line.split(",")]
        index = int(index)
        states[index] = {"free": int(free), "utilization": int(utilization)}
        uuid_to_index[uuid] = index

    busy = set()
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in apps.splitlines():
        if not line.strip():
            continue
        uuid = line.split(",", 1)[0].strip()
        if uuid in uuid_to_index:
            busy.add(uuid_to_index[uuid])
    return states, busy


def job_command(repo, baseline_data, view_root, output_root, video, group, gpu):
    output = os.path.join(output_root, video, group)
    if group == "coarse_qgsfm":
        dataset = os.path.join(baseline_data, video)
        command = ["bash", "scripts/train_xx.sh", dataset, output, str(gpu), "1"]
    elif group == "cfhs_qgsfm":
        dataset = os.path.join(view_root, video, group)
        command = ["bash", "scripts/train_xx.sh", dataset, output, str(gpu), "1"]
    else:
        dataset = os.path.join(view_root, video, group)
        command = ["bash", "scripts/train_uvw_pose.sh", dataset, output, str(gpu), "1"]
    return output, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--minimum-free-mib", type=int, default=24000)
    parser.add_argument("--maximum-utilization", type=int, default=20)
    parser.add_argument(
        "--allow-external-processes",
        action="store_true",
        help="Allow launching on a GPU that already has a compute process when memory permits",
    )
    parser.add_argument("--repo", default="/home/lzq/paperCode/talkinghead/TalkingGaussian")
    parser.add_argument(
        "--baseline-data",
        default="/home/lzq/paperCode/talkinghead/TalkingGaussian_baseline/data",
    )
    parser.add_argument(
        "--view-root", default="/mnt/folder_2/lzq/TalkingGaussian/ablation_data"
    )
    parser.add_argument(
        "--output-root", default="/mnt/folder_2/lzq/TalkingGaussian/ablation_output"
    )
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]

    jobs = [(video, group) for video in VIDEOS for group in GROUPS]
    pending = []
    for video, group in jobs:
        output = os.path.join(args.output_root, video, group)
        if os.path.isfile(os.path.join(output, ".ablation_complete")):
            print("[%s] skip completed %s/%s" % (timestamp(), video, group), flush=True)
        else:
            pending.append((video, group))

    running = {}
    env = os.environ.copy()
    env.update(POSE_START="6000", POSE_STOP="40000", POSE_INTERVAL="5")

    while pending or running:
        for gpu, task in list(running.items()):
            process, log_stream, video, group, output = task
            code = process.poll()
            if code is None:
                continue
            log_stream.close()
            marker = ".ablation_complete" if code == 0 else ".ablation_failed"
            with open(os.path.join(output, marker), "w") as stream:
                stream.write("%s exit_code=%d\n" % (timestamp(), code))
            running_marker = os.path.join(output, ".ablation_running")
            if os.path.exists(running_marker):
                os.remove(running_marker)
            running.pop(gpu)
            print(
                "[%s] finished GPU %d %s/%s exit=%d"
                % (timestamp(), gpu, video, group, code),
                flush=True,
            )

        if pending:
            try:
                states, externally_busy = gpu_state()
            except Exception as error:
                print("[%s] GPU query failed: %s" % (timestamp(), error), flush=True)
                time.sleep(args.poll_seconds)
                continue

            for gpu in gpus:
                if not pending or gpu in running:
                    continue
                if gpu in externally_busy and not args.allow_external_processes:
                    continue
                state = states.get(gpu)
                if (state is None
                        or state["free"] < args.minimum_free_mib
                        or state["utilization"] > args.maximum_utilization):
                    continue
                video, group = pending.pop(0)
                output, command = job_command(
                    args.repo,
                    args.baseline_data,
                    args.view_root,
                    args.output_root,
                    video,
                    group,
                    gpu,
                )
                os.makedirs(output, exist_ok=True)
                for marker in (".ablation_failed", ".ablation_running"):
                    path = os.path.join(output, marker)
                    if os.path.exists(path):
                        os.remove(path)
                with open(os.path.join(output, ".ablation_running"), "w") as stream:
                    stream.write("%s gpu=%d\n" % (timestamp(), gpu))
                log_stream = open(os.path.join(output, "train.log"), "a", buffering=1)
                log_stream.write("\n[%s] command: %s\n" % (timestamp(), " ".join(command)))
                process = subprocess.Popen(
                    command,
                    cwd=args.repo,
                    env=env,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                )
                running[gpu] = (process, log_stream, video, group, output)
                print(
                    "[%s] started GPU %d %s/%s pid=%d"
                    % (timestamp(), gpu, video, group, process.pid),
                    flush=True,
                )

        if pending or running:
            time.sleep(args.poll_seconds)

    print("[%s] all ablation jobs finished" % timestamp(), flush=True)


if __name__ == "__main__":
    main()
