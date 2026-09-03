import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.head_stabilizer.stabilizer import LPHSConfig, run_lphs


def main():
    parser = argparse.ArgumentParser(description="Large-Pose-Aware Head Stabilizer")
    parser.add_argument("--data", required=True, help="processed dataset directory")
    parser.add_argument("--flow_backend", choices=["raft", "dis"], default="raft")
    parser.add_argument("--preset", choices=["fast", "balanced", "quality"], default="balanced")
    parser.add_argument("--optimization_steps", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = LPHSConfig(
        flow_backend=args.flow_backend,
        preset=args.preset,
        optimization_steps=args.optimization_steps,
    )
    output, diagnostics = run_lphs(args.data, config, overwrite=args.overwrite)
    print("LPHS parameters saved to %s" % output)
    print("Before: %s" % diagnostics["before"])
    print("After: %s" % diagnostics["after"])


if __name__ == "__main__":
    main()
