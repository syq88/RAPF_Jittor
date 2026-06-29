import json
import os
import platform
import random
import statistics
import subprocess
import time

import hydra
import jittor as jt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Jittor 需要尽早设置 CUDA 标志；如果等到模型/nn 模块导入后再切换，部分环境会在 setter_use_cuda 崩溃。
jt.flags.use_cuda = int(os.environ.get("use_cuda", os.environ.get("USE_CUDA", 0)))

from continual_clip import utils
from continual_clip.jittor_datasets import build_cl_scenarios
from continual_clip.jittor_metrics import IncrementalMetricLogger
from continual_clip.jittor_models import load_model, sample
from jittor import nn


TRAIN_LOG = "train_log.csv"
PERF_LOG = "performance_log.json"
RESULT_SUMMARY = "result_summary.json"


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _old_class_batch_count(cfg):
    if cfg.dataset == "cifar100" and cfg.increment == 5:
        return 4
    return 2


def _make_synthetic_old_samples(cfg, model, task_id, batch_id):
    random_class_order_list = list(range(cfg.initial_increment + (task_id - 1) * cfg.increment))
    random.shuffle(random_class_order_list)
    per_batch = _old_class_batch_count(cfg)
    selected = [random_class_order_list[(batch_id * per_batch + i) % len(random_class_order_list)] for i in range(per_batch)]
    sg_inputs, sg_targets = [], []
    for class_id in selected:
        size = int(10 * cfg.beta)
        sg_inputs.append(sample(model.class_mean_list[class_id], model.class_cov_list[class_id], size, shrink=cfg.shrinkage))
        sg_targets.append(jt.ones((size,), dtype="int32") * class_id)
    return jt.concat(sg_inputs, dim=0), jt.concat(sg_targets, dim=0)


def _make_edge_samples(cfg, model):
    if not model.hard_pairs:
        return None, None, None
    edge_sample, edge_p_target, edge_n_target = [], [], []
    for positive, nearest in model.hard_pairs:
        size = int(20 * cfg.beta)
        edge_sample.append(sample(model.class_mean_list[positive], model.class_cov_list[positive], size, shrink=cfg.shrinkage))
        edge_p_target.append(jt.ones((size,), dtype="int32") * positive)
        edge_n_target.append(jt.ones((size,), dtype="int32") * nearest)
    return jt.concat(edge_sample, dim=0), jt.concat(edge_p_target, dim=0), jt.concat(edge_n_target, dim=0)


def _set_lr(optimizer, lr):
    optimizer.lr = lr


def _safe_float(var):
    arr = var.numpy()
    return float(arr.reshape(-1)[0])


def _append_train_log(row):
    exists = os.path.exists(TRAIN_LOG)
    with open(TRAIN_LOG, "a+") as f:
        if not exists:
            f.write("task,epoch,batch,loss,loss_c,loss_hinge,lr,batch_time_sec,samples\n")
        f.write(
            "{task},{epoch},{batch},{loss:.8f},{loss_c:.8f},{loss_hinge:.8f},{lr:.8f},{batch_time_sec:.6f},{samples}\n".format(
                **row
            )
        )


def _gpu_snapshot():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return output
    except Exception:
        return ""


def _write_performance_log(cfg, start_time, extra=None):
    # 记录训练环境和耗时，方便 README / PPT 中展示 Jittor 运行配置与性能信息。
    payload = {
        "elapsed_sec": round(time.perf_counter() - start_time, 4),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jittor": getattr(jt, "__version__", "unknown"),
        "use_cuda": int(jt.flags.use_cuda),
        "gpu": _gpu_snapshot(),
        "dataset": cfg.dataset,
        "config": {
            "initial_increment": int(cfg.initial_increment),
            "increment": int(cfg.increment),
            "epochs": int(cfg.epochs),
            "train_batch_size": int(cfg.train_batch_size),
            "batch_size": int(cfg.batch_size),
            "debug_max_batches": cfg.get("debug_max_batches"),
        },
    }
    if extra:
        payload.update(extra)
    with open(PERF_LOG, "w") as f:
        json.dump(payload, f, indent=2)


def run_class_incremental(cfg):
    run_start = time.perf_counter()
    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    model = load_model(cfg)
    train_dataset, classes_names = build_cl_scenarios(cfg, is_train=True)
    eval_dataset, _ = build_cl_scenarios(cfg, is_train=False)
    model.classes_names = classes_names
    metric_logger = IncrementalMetricLogger()
    acc_list = []
    if os.path.exists(TRAIN_LOG):
        os.remove(TRAIN_LOG)

    for task_id, _ in enumerate(eval_dataset):
        print(f"Train for task {task_id} has started.")
        model.adaptation(task_id, threshold=cfg.threshold)
        model.train()
        optimizer = nn.Adam(model.adapter.parameters(), lr=cfg.lr, weight_decay=0.0)

        for epoch in range(cfg.epochs):
            lr = cfg.lr * (0.1 ** sum(epoch >= milestone for milestone in cfg.milestones))
            _set_lr(optimizer, lr)
            loader = train_dataset[task_id].iter_batches(cfg.train_batch_size, shuffle=True)
            tqdm_loader = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
            for batch_id, (inputs, targets, task_ids) in enumerate(tqdm_loader):
                if cfg.get("debug_max_batches") is not None and batch_id >= cfg.debug_max_batches:
                    break
                batch_start = time.perf_counter()
                sg_inputs = None
                edge_sample = None
                edge_p_target = None
                edge_n_target = None
                if task_id > 0:
                    sg_inputs, sg_targets = _make_synthetic_old_samples(cfg, model, task_id, batch_id)
                    targets = jt.concat([targets, sg_targets], dim=0)
                edge_sample, edge_p_target, edge_n_target = _make_edge_samples(cfg, model)

                outputs, _, _, edge_sample_features = model(
                    inputs,
                    memory_data=sg_inputs,
                    not_ini=task_id > 0,
                    edge_sample=edge_sample,
                    prompt=False,
                )
                loss_c = nn.cross_entropy_loss(outputs, targets)
                loss_hinge = jt.array(0.0)
                if task_id > 0 and edge_sample is not None:
                    edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
                    edge_target_features = model.class_name_features[edge_p_target].float32()
                    edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)
                    edge_nearest_features = model.class_name_features[edge_n_target].float32()
                    edge_nearest_features = edge_nearest_features / edge_nearest_features.norm(dim=-1, keepdim=True)
                    # hard pair hinge loss 约束边界样本更接近正确旧类文本特征，而不是最近的新类文本特征。
                    pos = (edge_sample_features * edge_target_features.detach()).sum(dim=-1)
                    neg = (edge_sample_features * edge_nearest_features.detach()).sum(dim=-1)
                    loss_hinge = nn.relu(-pos + neg + 0.1).mean()
                loss = loss_c + loss_hinge
                optimizer.step(loss)
                loss_value = _safe_float(loss)
                loss_c_value = _safe_float(loss_c)
                loss_hinge_value = _safe_float(loss_hinge)
                _append_train_log(
                    {
                        "task": task_id,
                        "epoch": epoch + 1,
                        "batch": batch_id,
                        "loss": loss_value,
                        "loss_c": loss_c_value,
                        "loss_hinge": loss_hinge_value,
                        "lr": lr,
                        "batch_time_sec": time.perf_counter() - batch_start,
                        "samples": int(inputs.shape[0]),
                    }
                )
                tqdm_loader.set_description(
                    f"Epoch {epoch + 1}/{cfg.epochs} | loss {loss_value:.4f} | lr {lr:.5f}"
                )

        sample_data, sample_target = [], []
        print("Analyze feature distribution")
        for batch_id, (inputs, targets, task_ids) in enumerate(tqdm(train_dataset[task_id].iter_batches(128, shuffle=False))):
            if cfg.get("debug_max_batches") is not None and batch_id >= cfg.debug_max_batches:
                break
            with jt.no_grad():
                _, ori_ima_feat, _ = model(inputs, ori_ima_f=True)
            sample_data.append(ori_ima_feat)
            sample_target.append(targets)
        model.analyze_mean_cov(jt.concat(sample_data, dim=0), jt.concat(sample_target, dim=0))
        model.mix_matrix()

        model.eval()
        for batch_id, (inputs, targets, task_ids) in enumerate(eval_dataset[: task_id + 1].iter_batches(cfg.batch_size, shuffle=False)):
            if cfg.get("debug_max_batches") is not None and batch_id >= cfg.debug_max_batches:
                break
            with jt.no_grad():
                outputs, _, _, _ = model(inputs)
            preds = outputs.numpy().argmax(axis=1)
            metric_logger.add(preds, targets.numpy(), task_ids.numpy())

        acc_list.append(100 * metric_logger.accuracy)
        task_payload = {
            "task": task_id,
            "acc": round(100 * metric_logger.accuracy, 2),
            "avg_acc": round(100 * metric_logger.average_incremental_accuracy, 2),
            "forgetting": round(100 * metric_logger.forgetting, 6),
            "acc_per_task": [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
            "bwt": round(100 * metric_logger.backward_transfer, 2),
            "fwt": round(100 * metric_logger.forward_transfer, 2),
        }
        with open(cfg.log_path, "a+") as f:
            f.write(json.dumps(task_payload) + "\n")
        metric_logger.end_task()
        _write_performance_log(cfg, run_start, {"last_task": task_payload})

    final_payload = {"last": round(acc_list[-1], 2), "avg": round(statistics.mean(acc_list), 2)}
    with open(cfg.log_path, "a+") as f:
        f.write(json.dumps(final_payload) + "\n")
    with open(RESULT_SUMMARY, "w") as f:
        json.dump(final_payload, f, indent=2)
    _write_performance_log(cfg, run_start, {"final": final_payload})


@hydra.main(config_path=None, config_name=None, version_base="1.1")
def continual_clip_jittor(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    cfg_use_cuda = int(cfg.get("use_cuda", os.environ.get("use_cuda", jt.flags.use_cuda)))
    if cfg_use_cuda != int(jt.flags.use_cuda):
        # CUDA 最好通过命令前缀 use_cuda=1 在进程启动时打开，这里仅保留兼容入口。
        jt.flags.use_cuda = cfg_use_cuda
    cfg.workdir = utils.get_workdir(path=os.getcwd())
    if not cfg.dataset_root:
        # 默认把数据集放在项目目录下，符合当前迁移约定。
        cfg.dataset_root = os.path.join(cfg.workdir, "data", cfg.dataset)
    elif not os.path.isabs(cfg.dataset_root):
        cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)
    OmegaConf.save(cfg, "config_jittor.yaml")
    with open(cfg.log_path, "w+"):
        pass
    if cfg.scenario == "class":
        run_class_incremental(cfg)


if __name__ == "__main__":
    continual_clip_jittor()
