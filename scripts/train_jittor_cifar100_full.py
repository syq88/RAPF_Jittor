import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", type=int, default=int(os.environ.get("use_cuda", "1")))
    parser.add_argument("--data-root", default="data/cifar100")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "main_jittor.py",
        "--config-path",
        "configs/class",
        "--config-name",
        "cifar100_10-10",
        "class_order=class_orders/cifar100_order.yaml",
        f"dataset_root={args.data_root}",
        f"+use_cuda={args.use_cuda}",
    ]
    env = os.environ.copy()
    env["use_cuda"] = str(args.use_cuda)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
