import subprocess
import sys


def main():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements_jittor.txt",
            "-i",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
