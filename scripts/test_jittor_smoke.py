import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_test(path, env):
    subprocess.run([sys.executable, path], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", type=int, default=int(os.environ.get("use_cuda", "1")))
    args = parser.parse_args()
    env = os.environ.copy()
    env["use_cuda"] = str(args.use_cuda)
    run_test("tests/test_jittor_clip_vit_b16_real.py", env)
    run_test("tests/test_rapf_jittor_core.py", env)
    run_test("tests/test_rapf_jittor_smoke.py", env)


if __name__ == "__main__":
    main()
