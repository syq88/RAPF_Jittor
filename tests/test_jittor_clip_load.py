import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np

from continual_clip.jittor_clip import CLIP, load


def main():
    model = CLIP(
        embed_dim=32,
        image_resolution=32,
        vision_layers=1,
        vision_width=64,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=64,
        transformer_heads=1,
        transformer_layers=1,
    )
    state = {k: v.numpy() for k, v in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "clip-mini.npz")
        np.savez(path, **state)
        loaded, preprocess = load(path)
        assert loaded.visual.input_resolution == 32
        assert callable(preprocess)
    print("jittor_clip_load ok")


if __name__ == "__main__":
    main()
