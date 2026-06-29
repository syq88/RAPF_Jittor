import copy

import jittor as jt
from jittor import nn

from . import jittor_clip as clip
from .utils import get_class_ids_per_task, get_class_names


def _eye(size: int):
    return jt.init.eye((size, size), dtype="float32")


def _cdist(x, y):
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return (diff * diff).sum(dim=-1).sqrt()


def _cov(x):
    centered = x - x.mean(dim=0, keepdim=True)
    denom = max(int(x.shape[0]) - 1, 1)
    return centered.permute(1, 0) @ centered / denom


def shrink_cov(cov):
    diag = jt.diag(cov)
    diag_mean = diag.mean()
    iden = _eye(cov.shape[0])
    off_diag = cov * (1 - iden)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag * mask.float32()).sum() / jt.maximum(mask.float32().sum(), jt.array(1.0))
    return cov + diag_mean * iden + off_diag_mean * (1 - iden)


def sample(mean, cov, size, shrink=False):
    vec = jt.randn((size, mean.shape[-1]))
    if shrink:
        cov = shrink_cov(cov)
    # 生成旧类特征时使用 Cholesky，把标准高斯映射到估计的类协方差分布。
    sqrt_cov = jt.linalg.cholesky(cov)
    return vec @ sqrt_cov.permute(1, 0) + mean


class ClassIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device=None, jit=False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.classes_names = None
        model, self.transforms = clip.load(cfg.model_name)
        self.clip_model = model
        self.visual = model.visual
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.token_embedding = model.token_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dtype = "float32"
        self.adapter = nn.Linear(512, 512, bias=False)
        self.clip_type = model.dtype
        self.old_adapter = None
        self.class_mean_list = []
        self.class_cov_list = []
        self.class_edge_distance = []
        self.class_diff = None
        self.hard_pairs = []
        self.mix_b = cfg.mix_bias

    def encode_text(self, text, prompt=False):
        return self.clip_model.encode_text(text)

    def encode_image(self, image):
        return self.clip_model.encode_image(image)

    def get_class_name_features(self):
        with jt.no_grad():
            class_name_features = self.encode_text(self.text_tokens)
        return class_name_features.float32()

    def execute(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None, prompt=False):
        with jt.no_grad():
            text_features = self.encode_text(self.text_tokens)
            image_features = self.encode_image(image)
            original_image_features = image_features.detach()

        if memory_data is not None:
            image_features = jt.concat([image_features, memory_data.float32()], dim=0)
        edge_num = 0
        if edge_sample is not None:
            edge_num = edge_sample.shape[0]
            image_features = jt.concat([image_features, edge_sample.float32()], dim=0)

        # CLIP 主干冻结，只训练 adapter；detach 防止梯度回传进 backbone。
        image_features = self.adapter(image_features.float32().detach()).float32()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        edge_sample_features = None
        if edge_sample is not None:
            edge_sample_features = image_features[-edge_num:]
            image_features = image_features[:-edge_num]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits_per_image = self.logit_scale.exp() * image_features @ text_features.permute(1, 0)
        if not_ini:
            with jt.no_grad():
                old_memory_feature = self.old_adapter(memory_data.float32())
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return logits_per_image, image_features, old_memory_feature, edge_sample_features
            return logits_per_image, image_features, old_memory_feature, text_features
        if ori_ima_f:
            if memory_data is not None:
                image_features = image_features[:-memory_data.shape[0]]
            return logits_per_image, original_image_features, image_features
        return logits_per_image, image_features, None, None

    def adaptation(self, task_id, threshold=0):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize([self.prompt_template.format(c) for c in self.current_class_names])
        self.class_name_features = self.get_class_name_features()
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, keepdim=True)
        self.hard_pairs = []
        if task_id > 0:
            # 保存上一阶段 adapter，后续用于 RAPF 的参数融合和旧类特征约束。
            self.old_adapter = copy.deepcopy(self.adapter)
            old_num = self.cfg.initial_increment + (task_id - 1) * self.cfg.increment
            new_features = self.class_name_features[old_num:]
            old_features = self.class_name_features[:old_num]
            dist = _cdist(old_features.float32(), new_features.float32())
            self.class_diff = dist
            indices = jt.nonzero(dist < threshold).numpy()
            offset = old_num
            self.hard_pairs = [(int(old_idx), int(new_idx + offset)) for old_idx, new_idx in indices]

    def analyze_mean_cov(self, features, labels):
        labels_np = labels.numpy().astype("int32")
        for label in sorted(np_unique(labels_np)):
            idx = np_where(labels_np == label)
            class_data = features[idx]
            mean = class_data.mean(dim=0)
            cov = _cov(class_data.float32()) + 1e-4 * _eye(class_data.shape[-1])
            distance = _cdist(class_data.float32(), mean.unsqueeze(0)).squeeze()
            sorted_distance = jt.sort(distance)[0]
            max_distance = sorted_distance[-min(10, sorted_distance.shape[0]) :]
            self.class_edge_distance.append((max_distance.mean(), max_distance.max()))
            self.class_mean_list.append(mean.detach())
            self.class_cov_list.append(cov.detach())

    def mix_matrix(self):
        if self.old_adapter is None:
            return
        weight_new = self.adapter.weight
        weight_old = self.old_adapter.weight
        # RAPF 的参数融合：在旧 adapter 的 SVD 坐标系下，只让变化大的方向更多采用新参数。
        U_old, S_old, V_old = jt.linalg.svd(weight_old)
        old_right = jt.diag(S_old) @ V_old
        P_new = U_old.permute(1, 0) @ weight_new
        dist = (P_new - old_right).abs()
        mask = dist / jt.maximum(dist.max(), jt.array(1e-12))
        mask = jt.clamp(mask + self.mix_b, max_v=1)
        right = P_new * mask + old_right * (1 - mask)
        self.adapter.weight.assign(U_old @ right)


def np_unique(x):
    import numpy as np
    return np.unique(x)


def np_where(mask):
    import numpy as np
    return np.where(mask)[0]


def load_model(cfg, device=None) -> nn.Module:
    if cfg.scenario == "class":
        return ClassIncrementalCLIP(cfg, device)
    raise NotImplementedError("Jittor v1 先支持 class-incremental RAPF。")
