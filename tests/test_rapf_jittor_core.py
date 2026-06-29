import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import jittor as jt

from continual_clip.jittor_models import sample, shrink_cov


def main():
    mean = jt.zeros((4,))
    cov = jt.init.eye((4, 4), dtype="float32")
    draws = sample(mean, cov, 8, shrink=True)
    assert tuple(draws.shape) == (8, 4)
    shrunk = shrink_cov(cov)
    assert tuple(shrunk.shape) == (4, 4)
    print("rapf_jittor_core ok")


if __name__ == "__main__":
    main()
