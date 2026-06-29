import os
import pickle
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import jittor as jt
import numpy as np
from PIL import Image
from tqdm import tqdm

from .jittor_clip import preprocess_image


CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_TAR = "cifar-100-python.tar.gz"


def _download(url: str, target: str):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target):
        return
    with urllib.request.urlopen(url) as source, open(target, "wb") as output:
        total = int(source.info().get("Content-Length", 0))
        with tqdm(total=total, unit="iB", unit_scale=True, ncols=80) as loop:
            while True:
                chunk = source.read(8192)
                if not chunk:
                    break
                output.write(chunk)
                loop.update(len(chunk))


def ensure_cifar100(root: str):
    os.makedirs(root, exist_ok=True)
    extracted = os.path.join(root, "cifar-100-python")
    if os.path.isdir(extracted):
        return extracted
    tar_path = os.path.join(root, CIFAR100_TAR)
    _download(CIFAR100_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(root)
    return extracted


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_cifar100(root: str, train: bool):
    base = ensure_cifar100(root)
    data = _load_pickle(os.path.join(base, "train" if train else "test"))
    meta = _load_pickle(os.path.join(base, "meta"))
    images = data["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(data["fine_labels"], dtype=np.int32)
    return images, labels, list(meta["fine_label_names"])


@dataclass
class JittorTaskDataset:
    images: np.ndarray
    labels: np.ndarray
    task_ids: np.ndarray
    classes: Sequence[int]
    image_size: int = 224

    def __len__(self):
        return int(self.labels.shape[0])

    def iter_batches(self, batch_size: int, shuffle: bool = False):
        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_images = []
            for idx in batch_indices:
                image = Image.fromarray(self.images[idx], mode="RGB")
                batch_images.append(preprocess_image(image, self.image_size).numpy())
            # 图像预处理在 numpy/PIL 侧完成，进入模型前一次性转成 jt.Var，避免每张图单独触发同步。
            inputs = jt.array(np.stack(batch_images, axis=0).astype("float32"))
            targets = jt.array(self.labels[batch_indices].astype("int32"))
            task_ids = jt.array(self.task_ids[batch_indices].astype("int32"))
            yield inputs, targets, task_ids


class ClassIncrementalScenario:
    def __init__(self, tasks: List[JittorTaskDataset]):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __getitem__(self, item):
        if isinstance(item, slice):
            selected = self.tasks[item]
            images = np.concatenate([task.images for task in selected], axis=0)
            labels = np.concatenate([task.labels for task in selected], axis=0)
            task_ids = np.concatenate([task.task_ids for task in selected], axis=0)
            classes = []
            for task in selected:
                classes.extend(task.classes)
            return JittorTaskDataset(images, labels, task_ids, classes)
        return self.tasks[item]


def _make_tasks(images: np.ndarray, labels: np.ndarray, class_order: Sequence[int], initial_increment: int, increment: int):
    task_classes = [list(class_order[:initial_increment])]
    for start in range(initial_increment, len(class_order), increment):
        task_classes.append(list(class_order[start : start + increment]))

    label_to_incremental = {int(class_id): idx for idx, class_id in enumerate(class_order)}
    tasks = []
    for task_id, classes in enumerate(task_classes):
        mask = np.isin(labels, np.asarray(classes, dtype=np.int32))
        task_images = images[mask]
        # CLIP logits 的列顺序是 class_order 的增量顺序，不是 CIFAR 原始 label。
        # 因此这里必须把原始 CIFAR label 映射成 [0, num_seen_classes) 的训练目标。
        task_labels = np.asarray([label_to_incremental[int(label)] for label in labels[mask]], dtype=np.int32)
        task_ids = np.full(task_labels.shape, task_id, dtype=np.int32)
        tasks.append(JittorTaskDataset(task_images, task_labels, task_ids, classes))
    return tasks


def build_cl_scenarios(cfg, is_train: bool):
    if cfg.dataset != "cifar100":
        raise NotImplementedError("Jittor v1 数据管线先支持 cifar100，其他数据集后续再迁移。")
    images, labels, class_names = load_cifar100(cfg.dataset_root, train=is_train)
    tasks = _make_tasks(images, labels, cfg.class_order, cfg.initial_increment, cfg.increment)
    return ClassIncrementalScenario(tasks), class_names
