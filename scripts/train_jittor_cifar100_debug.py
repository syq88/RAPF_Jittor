import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", type=int, default=int(os.environ.get("use_cuda", "1")))
    parser.add_argument("--data-root", default="data/cifar100")
    parser.add_argument("--output-dir", default="./experiments/debug_jittor_cifar")
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
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
        f"epochs={args.epochs}",
        f"train_batch_size={args.batch_size}",
        f"batch_size={args.batch_size}",
        f"+debug_max_batches={args.max_batches}",
        f"+use_cuda={args.use_cuda}",
        f"hydra.run.dir={args.output_dir}",
    ]
    env = os.environ.copy()
    env["use_cuda"] = str(args.use_cuda)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
