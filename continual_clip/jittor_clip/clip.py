import hashlib
import os
import urllib.request
from typing import List, Union

import jittor as jt
import numpy as np
from PIL import Image
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer

__all__ = ["available_models", "load", "preprocess_image", "tokenize"]

_tokenizer = SimpleTokenizer()

_MODELS = {
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

_NPZ_NAMES = {
    "ViT-B/16": "ViT-B-16-jittor.npz",
}


def available_models() -> List[str]:
    return list(_MODELS.keys())


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)
    if os.path.isfile(download_target) and hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        total = int(source.info().get("Content-Length"))
        with tqdm(total=total, ncols=80, unit="iB", unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Downloaded CLIP checkpoint has an invalid SHA256 checksum.")
    return download_target


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int32)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = np.asarray(tokens, dtype=np.int32)

    # tokenizer 仍用 Python/BPE，最后一步才转成 jt.Var，保持和 OpenAI CLIP 的 token id 完全一致。
    return jt.array(result)


def preprocess_image(image: Image.Image, n_px: int = 224):
    image = image.convert("RGB")
    image = image.resize((n_px, n_px), Image.BICUBIC)
    arr = np.asarray(image).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    mean = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype="float32").reshape(3, 1, 1)
    std = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype="float32").reshape(3, 1, 1)
    return jt.array((arr - mean) / std)


def _load_npz(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load(name: str, device=None, jit: bool = False, download_root: str = None, checkpoint_path: str = None):
    if jit:
        raise NotImplementedError("Jittor CLIP does not support torchscript jit checkpoints.")
    if device is not None:
        # Jittor 通过 jt.flags.use_cuda 控制设备；保留参数只是为了兼容原 clip.load 调用。
        pass

    if checkpoint_path is None:
        if os.path.isfile(name):
            checkpoint_path = name
        elif name in _NPZ_NAMES:
            root = download_root or os.path.expanduser("~/.cache/clip")
            checkpoint_path = os.path.join(root, _NPZ_NAMES[name])
        else:
            raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Jittor CLIP checkpoint not found: {checkpoint_path}. "
            "Run tools/convert_openai_clip_to_jittor.py to convert the OpenAI .pt checkpoint first."
        )

    state_dict = _load_npz(checkpoint_path)
    model = build_model(state_dict)
    resolution = model.visual.input_resolution
    return model, lambda image: preprocess_image(image, resolution)
