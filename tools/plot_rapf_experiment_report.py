import argparse
import csv
import glob
import json
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing plotting dependencies. Install them with "
        "`pip install numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple`."
    ) from exc


def latest(pattern):
    candidates = [path for path in glob.glob(pattern) if not path.endswith(":Zone.Identifier")]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_jsonl_metric(path):
    rows, summary = [], {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "task" in item:
                rows.append(item)
            else:
                summary = item
    return rows, summary


def load_train_log(path):
    rows = []
    with open(path, "r") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            parsed = {
                "step": idx,
                "task": int(row["task"]),
                "epoch": int(row["epoch"]),
                "batch": int(row["batch"]),
                "loss": float(row["loss"]),
                "loss_c": float(row["loss_c"]),
                "loss_hinge": float(row["loss_hinge"]),
                "lr": float(row["lr"]),
                "batch_time_sec": float(row["batch_time_sec"]),
                "samples": int(row["samples"]),
            }
            rows.append(parsed)
    return rows


def smooth(values, window):
    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(np.asarray(values, dtype=np.float32), kernel, mode="same")


def plot_metric_curve(pt_rows, jt_rows, key, ylabel, title, output):
    plt.figure(figsize=(8, 4.8))
    plt.plot([r["task"] for r in pt_rows], [r[key] for r in pt_rows], marker="o", label="PyTorch")
    plt.plot([r["task"] for r in jt_rows], [r[key] for r in jt_rows], marker="s", label="Jittor")
    plt.xlabel("Task")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def plot_metric_delta(pt_rows, jt_rows, output):
    tasks = [r["task"] for r in pt_rows]
    delta = [jt["acc"] - pt["acc"] for pt, jt in zip(pt_rows, jt_rows)]
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in delta]
    plt.figure(figsize=(8, 4.2))
    plt.bar(tasks, delta, color=colors, alpha=0.85)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Task")
    plt.ylabel("Jittor - PyTorch Acc (%)")
    plt.title("Per-task Accuracy Difference")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def plot_final_bar(pt_summary, jt_summary, output):
    labels = ["Last", "Average"]
    pt = [pt_summary["last"], pt_summary["avg"]]
    jt = [jt_summary["last"], jt_summary["avg"]]
    x = np.arange(len(labels))
    width = 0.34
    plt.figure(figsize=(6.4, 4.4))
    plt.bar(x - width / 2, pt, width, label="PyTorch")
    plt.bar(x + width / 2, jt, width, label="Jittor")
    for xs, values in [(x - width / 2, pt), (x + width / 2, jt)]:
        for x_i, value in zip(xs, values):
            plt.text(x_i, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Final Result Comparison")
    plt.ylim(0, max(pt + jt) + 8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def plot_heatmap(rows, title, output):
    max_len = max(len(row.get("acc_per_task", [])) for row in rows)
    matrix = np.full((len(rows), max_len), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        values = row.get("acc_per_task", [])
        matrix[i, : len(values)] = values
    plt.figure(figsize=(7.2, 5.8))
    image = plt.imshow(np.ma.masked_invalid(matrix), cmap="viridis", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(image, label="Accuracy (%)")
    plt.xlabel("Evaluated Task")
    plt.ylabel("After Training Task")
    plt.title(title)
    plt.xticks(range(max_len))
    plt.yticks(range(len(rows)))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                plt.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white")
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def _acc_matrix(rows):
    max_len = max(len(row.get("acc_per_task", [])) for row in rows)
    matrix = np.full((len(rows), max_len), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        values = row.get("acc_per_task", [])
        matrix[i, : len(values)] = values
    return matrix


def plot_heatmap_delta(pt_rows, jt_rows, output):
    pt_matrix = _acc_matrix(pt_rows)
    jt_matrix = _acc_matrix(jt_rows)
    rows = min(pt_matrix.shape[0], jt_matrix.shape[0])
    cols = min(pt_matrix.shape[1], jt_matrix.shape[1])
    delta = jt_matrix[:rows, :cols] - pt_matrix[:rows, :cols]
    max_abs = np.nanmax(np.abs(delta))
    max_abs = max(float(max_abs), 1.0)

    plt.figure(figsize=(7.4, 5.8))
    image = plt.imshow(np.ma.masked_invalid(delta), cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
    plt.colorbar(image, label="Jittor - PyTorch Accuracy (%)")
    plt.xlabel("Evaluated Task")
    plt.ylabel("After Training Task")
    plt.title("Per-task Accuracy Difference Heatmap")
    plt.xticks(range(cols))
    plt.yticks(range(rows))
    for i in range(rows):
        for j in range(cols):
            if not np.isnan(delta[i, j]):
                color = "white" if abs(delta[i, j]) > max_abs * 0.45 else "black"
                plt.text(j, i, f"{delta[i, j]:+.1f}", ha="center", va="center", fontsize=7, color=color)
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def plot_loss_comparison(pt_log, jt_log, key, ylabel, title, output, window):
    plt.figure(figsize=(9, 4.8))
    for label, rows in [("PyTorch", pt_log), ("Jittor", jt_log)]:
        steps = [row["step"] for row in rows]
        values = smooth([row[key] for row in rows], window)
        plt.plot(steps, values, label=label, linewidth=1.4)
    plt.xlabel("Optimization Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def plot_task_loss_bands(rows, prefix, output_dir, window):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    plt.figure(figsize=(9, 4.8))
    for task, task_rows in sorted(by_task.items()):
        x = list(range(len(task_rows)))
        y = smooth([row["loss"] for row in task_rows], window)
        plt.plot(x, y, linewidth=1.2, label=f"Task {task}")
    plt.xlabel("Step Within Task")
    plt.ylabel("Loss")
    plt.title(f"{prefix} Loss by Task")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix.lower()}_loss_by_task.png"), dpi=220)
    plt.close()


def plot_batch_time(pt_log, jt_log, output, window):
    plt.figure(figsize=(9, 4.4))
    for label, rows in [("PyTorch", pt_log), ("Jittor", jt_log)]:
        steps = [row["step"] for row in rows]
        values = smooth([row["batch_time_sec"] for row in rows], window)
        plt.plot(steps, values, linewidth=1.2, label=label)
    plt.xlabel("Optimization Step")
    plt.ylabel("Batch Time (sec)")
    plt.title("Batch Time Comparison")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def write_performance_summary(exp_dir, pt_summary, jt_summary, output):
    pt_perf = json.load(open(os.path.join(exp_dir, "performance_log.json")))
    jt_perf_path = os.path.join(exp_dir, "performance_log_jittor.json")
    jt_perf = json.load(open(jt_perf_path)) if os.path.exists(jt_perf_path) else {}
    with open(output, "w") as f:
        f.write("item,pytorch,jittor\n")
        f.write(f"last_acc,{pt_summary.get('last','')},{jt_summary.get('last','')}\n")
        f.write(f"avg_acc,{pt_summary.get('avg','')},{jt_summary.get('avg','')}\n")
        f.write(f"elapsed_sec,{pt_perf.get('elapsed_sec','')},{jt_perf.get('elapsed_sec','')}\n")
        f.write(f"gpu,\"{pt_perf.get('gpu','')}\",\"{jt_perf.get('gpu','')}\"\n")
        f.write(f"framework,\"torch {pt_perf.get('torch','')}\",\"jittor {jt_perf.get('jittor','')}\"\n")
        f.write(f"python,{pt_perf.get('python','')},{jt_perf.get('python','')}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="experiments/class/cifar100_10-10_without_exp")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.exp_dir, "figures_full")
    os.makedirs(output_dir, exist_ok=True)

    pt_metric = os.path.join(args.exp_dir, "metric.json")
    jt_metric = latest(os.path.join(args.exp_dir, "metric_jittor*.json"))
    if jt_metric is None:
        raise FileNotFoundError("No metric_jittor*.json found.")
    pt_rows, pt_summary = load_jsonl_metric(pt_metric)
    jt_rows, jt_summary = load_jsonl_metric(jt_metric)

    for key, ylabel, title, filename in [
        ("acc", "Accuracy (%)", "Incremental Accuracy: PyTorch vs Jittor", "acc_curve.png"),
        ("avg_acc", "Average Accuracy (%)", "Average Incremental Accuracy: PyTorch vs Jittor", "avg_acc_curve.png"),
        ("forgetting", "Forgetting (%)", "Forgetting: PyTorch vs Jittor", "forgetting_curve.png"),
        ("bwt", "Backward Transfer (%)", "Backward Transfer: PyTorch vs Jittor", "bwt_curve.png"),
    ]:
        plot_metric_curve(pt_rows, jt_rows, key, ylabel, title, os.path.join(output_dir, filename))
    plot_metric_delta(pt_rows, jt_rows, os.path.join(output_dir, "acc_delta.png"))
    plot_final_bar(pt_summary, jt_summary, os.path.join(output_dir, "final_comparison.png"))
    plot_heatmap(pt_rows, "Per-task Accuracy Heatmap (PyTorch)", os.path.join(output_dir, "acc_per_task_heatmap_pytorch.png"))
    plot_heatmap(jt_rows, "Per-task Accuracy Heatmap (Jittor)", os.path.join(output_dir, "acc_per_task_heatmap_jittor.png"))
    plot_heatmap_delta(pt_rows, jt_rows, os.path.join(output_dir, "acc_per_task_heatmap_delta.png"))

    pt_train = os.path.join(args.exp_dir, "train_log.csv")
    jt_train = os.path.join(args.exp_dir, "train_log_jittor.csv")
    if os.path.exists(pt_train) and os.path.exists(jt_train):
        pt_log = load_train_log(pt_train)
        jt_log = load_train_log(jt_train)
        plot_loss_comparison(pt_log, jt_log, "loss", "Loss", "Training Loss: PyTorch vs Jittor", os.path.join(output_dir, "loss_curve.png"), args.smooth_window)
        plot_loss_comparison(pt_log, jt_log, "loss_c", "Classification Loss", "Classification Loss: PyTorch vs Jittor", os.path.join(output_dir, "loss_c_curve.png"), args.smooth_window)
        plot_loss_comparison(pt_log, jt_log, "loss_hinge", "Hinge Loss", "Hinge Loss: PyTorch vs Jittor", os.path.join(output_dir, "loss_hinge_curve.png"), args.smooth_window)
        plot_task_loss_bands(pt_log, "PyTorch", output_dir, args.smooth_window)
        plot_task_loss_bands(jt_log, "Jittor", output_dir, args.smooth_window)
        plot_batch_time(pt_log, jt_log, os.path.join(output_dir, "batch_time_curve.png"), args.smooth_window)

    write_performance_summary(args.exp_dir, pt_summary, jt_summary, os.path.join(output_dir, "performance_summary.csv"))
    print(f"Used Jittor metric: {jt_metric}")
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
