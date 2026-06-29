import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import jittor as jt

from continual_clip.jittor_clip import CLIP, tokenize


def main():
    tokens = tokenize(["a good photo of a dog.", "a good photo of a cat."])
    assert tuple(tokens.shape) == (2, 77)

    # 使用缩小版 CLIP 做快速 smoke test，避免完整 ViT-B/16 首次 JIT 编译耗时过长。
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
    image = jt.randn((2, 3, 32, 32))
    logits_per_image, logits_per_text = model(image, tokens)
    assert tuple(model.encode_image(image).shape) == (2, 32)
    assert tuple(model.encode_text(tokens).shape) == (2, 32)
    assert tuple(logits_per_image.shape) == (2, 2)
    assert tuple(logits_per_text.shape) == (2, 2)
    print("jittor_clip_smoke ok")


if __name__ == "__main__":
    main()
