import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from continual_clip.jittor_datasets import ensure_cifar100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data", "cifar100"))
    args = parser.parse_args()
    print(ensure_cifar100(args.data_dir))


if __name__ == "__main__":
    main()
