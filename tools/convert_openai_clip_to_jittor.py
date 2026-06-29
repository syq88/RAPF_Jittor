import argparse
import hashlib
import os
import urllib.request

import numpy as np
from tqdm import tqdm


_MODELS = {
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


def _split_qkv(state, old_prefix, new_prefix):
    weight = state.pop(f"{old_prefix}.attn.in_proj_weight")
    bias = state.pop(f"{old_prefix}.attn.in_proj_bias")
    q_w, k_w, v_w = np.split(weight, 3, axis=0)
    q_b, k_b, v_b = np.split(bias, 3, axis=0)
    state[f"{new_prefix}.attn.q_proj.weight"] = q_w
    state[f"{new_prefix}.attn.k_proj.weight"] = k_w
    state[f"{new_prefix}.attn.v_proj.weight"] = v_w
    state[f"{new_prefix}.attn.q_proj.bias"] = q_b
    state[f"{new_prefix}.attn.k_proj.bias"] = k_b
    state[f"{new_prefix}.attn.v_proj.bias"] = v_b
    state[f"{new_prefix}.attn.out_proj.weight"] = state.pop(f"{old_prefix}.attn.out_proj.weight")
    state[f"{new_prefix}.attn.out_proj.bias"] = state.pop(f"{old_prefix}.attn.out_proj.bias")


def download_checkpoint(model_name, download_root):
    if model_name not in _MODELS:
        raise RuntimeError(f"Unknown model {model_name}; available models: {sorted(_MODELS)}")

    url = _MODELS[model_name]
    os.makedirs(download_root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    target = os.path.join(download_root, filename)

    if os.path.isfile(target):
        actual_sha256 = hashlib.sha256(open(target, "rb").read()).hexdigest()
        if actual_sha256 == expected_sha256:
            return target
        print(f"Existing checkpoint checksum mismatch, re-downloading: {target}")

    with urllib.request.urlopen(url) as source, open(target, "wb") as output:
        total = int(source.info().get("Content-Length"))
        with tqdm(total=total, ncols=80, unit="iB", unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    actual_sha256 = hashlib.sha256(open(target, "rb").read()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError("Downloaded checkpoint has an invalid SHA256 checksum.")
    return target


def convert(input_path, output_path):
    # 这个脚本是离线权重转换工具，可以依赖 PyTorch；RAPF/Jittor 推理路径不再依赖 PyTorch。
    import torch

    with open(input_path, "rb") as opened_file:
        try:
            raw = torch.jit.load(opened_file, map_location="cpu").eval()
            state = raw.state_dict()
        except RuntimeError:
            opened_file.seek(0)
            raw = torch.load(opened_file, map_location="cpu")
            state = raw.state_dict() if hasattr(raw, "state_dict") else raw
    state = {
        k: v.detach().cpu().float().numpy()
        for k, v in state.items()
        if hasattr(v, "detach")
    }

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state.pop(key, None)

    prefixes = []
    for key in list(state.keys()):
        if key.endswith(".attn.in_proj_weight"):
            prefixes.append(key[: -len(".attn.in_proj_weight")])
    for prefix in prefixes:
        _split_qkv(state, prefix, prefix)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(output_path, **state)
    print(f"Saved Jittor CLIP checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="OpenAI CLIP .pt checkpoint path")
    source.add_argument("--model", choices=sorted(_MODELS), help="Download and convert an official OpenAI CLIP checkpoint")
    parser.add_argument("--output", required=True, help="Output .npz path for Jittor CLIP")
    parser.add_argument("--download-root", default=os.path.expanduser("~/.cache/clip"), help="Where to cache downloaded .pt checkpoints")
    args = parser.parse_args()
    input_path = args.input or download_checkpoint(args.model, args.download_root)
    convert(input_path, args.output)


if __name__ == "__main__":
    main()
