from typing import Tuple, Union

import numpy as np
import jittor as jt
from jittor import nn


class LayerNorm(nn.LayerNorm):
    def execute(self, x):
        # CLIP 在 fp16 下会把 LayerNorm 临时升到 fp32，避免归一化时数值不稳定。
        orig_dtype = x.dtype
        x = super().execute(x.float32())
        return x.astype(orig_dtype)


class QuickGELU(nn.Module):
    def execute(self, x):
        return x * jt.sigmoid(1.702 * x)


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)

    def execute(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _shape(self, x):
        seq_len, batch, _ = x.shape
        # 输入沿用 OpenAI CLIP 的 LND 布局；attention 计算时转成 N,H,L,D。
        return x.reshape(seq_len, batch, self.n_head, self.head_dim).permute(1, 2, 0, 3)

    def execute(self, x, attn_mask=None):
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        scale = self.head_dim ** -0.5
        attn = (q * scale) @ k.permute(0, 1, 3, 2)
        if attn_mask is not None:
            # 文本 Transformer 使用 causal mask；mask 形状为 [L,L]，可广播到 [N,H,L,L]。
            attn = attn + attn_mask.reshape(1, 1, attn_mask.shape[0], attn_mask.shape[1]).astype(attn.dtype)
        attn = nn.softmax(attn, dim=-1)
        out = attn @ v
        out = out.permute(2, 0, 1, 3).reshape(x.shape[0], x.shape[1], self.d_model)
        return self.out_proj(out)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.ln_2 = LayerNorm(d_model)
        self._attn_mask = attn_mask

    def attention(self, x):
        return self.attn(x, self._attn_mask)

    def execute(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def execute(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = scale * jt.randn(width)
        self.positional_embedding = scale * jt.randn((input_resolution // patch_size) ** 2 + 1, width)
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = scale * jt.randn(width, output_dim)

    def execute(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_embedding = self.class_embedding.astype(x.dtype).reshape(1, 1, -1)
        # class token 需要按 batch 复制；这里不用 numpy，避免断开 Jittor 计算图。
        class_tokens = jt.ones((x.shape[0], 1, x.shape[-1]), dtype=x.dtype) * class_embedding
        x = jt.concat([class_tokens, x], dim=1)
        x = x + self.positional_embedding.astype(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()
        if isinstance(vision_layers, (tuple, list)):
            raise NotImplementedError("Jittor CLIP v1 only supports ViT backbones, e.g. ViT-B/16.")

        self.context_length = context_length
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
        )
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = jt.empty((self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = jt.empty((transformer_width, embed_dim))
        self.logit_scale = jt.array(np.log(1 / 0.07)).float32()
        self.initialize_parameters()

    def initialize_parameters(self):
        self.token_embedding.weight.assign(jt.randn(self.token_embedding.weight.shape) * 0.02)
        self.positional_embedding.assign(jt.randn(self.positional_embedding.shape) * 0.01)
        self.text_projection.assign(jt.randn(self.text_projection.shape) * (self.transformer.width ** -0.5))

    def build_attention_mask(self):
        # OpenAI CLIP 的文本塔是 causal attention，当前位置只能看见自己和之前的 token。
        mask = np.full((self.context_length, self.context_length), -np.inf, dtype=np.float32)
        mask = np.triu(mask, k=1)
        return jt.array(mask)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.astype(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).astype(self.dtype)
        x = x + self.positional_embedding.astype(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).astype(self.dtype)

        # EOT token 在 OpenAI CLIP 词表中 id 最大；RAPF 的类别文本特征也依赖这个位置。
        eot_indices = text.argmax(dim=-1)[0]
        # Jittor 的高级索引会比 PyTorch 更容易发生广播；用 one-hot mask 精确取每条文本的 EOT 特征。
        eot_mask = (jt.arange(x.shape[1]).reshape(1, -1) == eot_indices.reshape(-1, 1)).float32()
        x = (x * eot_mask.unsqueeze(-1)).sum(dim=1) @ self.text_projection
        return x

    def execute(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits_per_image = (self.logit_scale.exp() * image_features) @ text_features.permute(1, 0)
        return logits_per_image, logits_per_image.permute(1, 0)


def build_model(state_dict: dict):
    if "visual.proj" not in state_dict:
        raise NotImplementedError("Jittor CLIP v1 only supports ViT checkpoints.")

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.q_proj.weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )
    model.load_state_dict({k: jt.array(v) for k, v in state_dict.items()})
    model.eval()
    return model
