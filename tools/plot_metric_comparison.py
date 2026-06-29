import argparse
import glob
import json
import math
import os
from typing import Dict, List, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: numpy. Install it with "
        "`pip install numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple`."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install it with "
        "`pip install numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple`."
    ) from exc


def load_metric_jsonl(path: str) -> Tuple[List[Dict], Dict]:
    task_rows = []
    summary = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "task" in row:
                task_rows.append(row)
            else:
                summary = row
    if not task_rows:
        raise ValueError(f"No task metrics found in {path}")
    return task_rows, summary


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_line_plot(
    pytorch_rows: List[Dict],
    jittor_rows: List[Dict],
    key: str,
    ylabel: str,
    title: str,
    output_path: str,
):
    pt_tasks = [row["task"] for row in pytorch_rows if key in row]
    pt_values = [row[key] for row in pytorch_rows if key in row]
    jt_tasks = [row["task"] for row in jittor_rows if key in row]
    jt_values = [row[key] for row in jittor_rows if key in row]

    plt.figure(figsize=(8, 4.8))
    plt.plot(pt_tasks, pt_values, marker="o", linewidth=2, label="PyTorch")
    plt.plot(jt_tasks, jt_values, marker="s", linewidth=2, label="Jittor")
    plt.xlabel("Task")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_delta_plot(pytorch_rows: List[Dict], jittor_rows: List[Dict], key: str, output_path: str):
    values = []
    tasks = []
    for pt_row, jt_row in zip(pytorch_rows, jittor_rows):
        if key in pt_row and key in jt_row:
            tasks.append(pt_row["task"])
            values.append(jt_row[key] - pt_row[key])

    plt.figure(figsize=(8, 4.2))
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]
    plt.bar(tasks, values, color=colors, alpha=0.82)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Task")
    plt.ylabel(f"Jittor - PyTorch {key}")
    plt.title(f"Per-task {key} Difference")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _acc_matrix(rows: List[Dict]) -> np.ndarray:
    max_len = max(len(row.get("acc_per_task", [])) for row in rows)
    matrix = np.full((len(rows), max_len), np.nan, dtype=np.float32)
    for row_id, row in enumerate(rows):
        values = row.get("acc_per_task", [])
        if values:
            matrix[row_id, : len(values)] = values
    return matrix


def save_heatmap(rows: List[Dict], label: str, output_path: str):
    matrix = _acc_matrix(rows)
    masked = np.ma.masked_invalid(matrix)

    plt.figure(figsize=(7.2, 5.8))
    image = plt.imshow(masked, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(image, label="Accuracy (%)")
    plt.xlabel("Evaluated Task")
    plt.ylabel("After Training Task")
    plt.title(f"Per-task Accuracy Heatmap ({label})")
    plt.xticks(range(matrix.shape[1]))
    plt.yticks(range(matrix.shape[0]))

    # 对热力图中的有效格子标数值，方便 README/PPT 直接展示。
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not math.isnan(float(matrix[i, j])):
                plt.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_final_bar(pt_summary: Dict, jt_summary: Dict, output_path: str):
    labels = ["Last Acc", "Avg Acc"]
    pt_values = [pt_summary.get("last", np.nan), pt_summary.get("avg", np.nan)]
    jt_values = [jt_summary.get("last", np.nan), jt_summary.get("avg", np.nan)]
    x = np.arange(len(labels))
    width = 0.34

    plt.figure(figsize=(6.4, 4.4))
    plt.bar(x - width / 2, pt_values, width, label="PyTorch")
    plt.bar(x + width / 2, jt_values, width, label="Jittor")
    for xs, values in [(x - width / 2, pt_values), (x + width / 2, jt_values)]:
        for x_i, value in zip(xs, values):
            if not np.isnan(value):
                plt.text(x_i, value + 0.4, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Final Result Comparison")
    plt.ylim(0, max(pt_values + jt_values) + 8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_summary_table(pt_rows: List[Dict], jt_rows: List[Dict], pt_summary: Dict, jt_summary: Dict, path: str):
    keys = ["acc", "avg_acc", "forgetting", "bwt", "fwt"]
    with open(path, "w") as f:
        f.write("task," + ",".join([f"pytorch_{key},jittor_{key},delta_{key}" for key in keys]) + "\n")
        for pt_row, jt_row in zip(pt_rows, jt_rows):
            cells = [str(pt_row["task"])]
            for key in keys:
                pt_value = float(pt_row.get(key, np.nan))
                jt_value = float(jt_row.get(key, np.nan))
                cells.extend([f"{pt_value:.6f}", f"{jt_value:.6f}", f"{jt_value - pt_value:.6f}"])
            f.write(",".join(cells) + "\n")
        f.write(f"summary_last,{pt_summary.get('last', '')},{jt_summary.get('last', '')},")
        if "last" in pt_summary and "last" in jt_summary:
            f.write(f"{jt_summary['last'] - pt_summary['last']:.6f}")
        f.write("\n")
        f.write(f"summary_avg,{pt_summary.get('avg', '')},{jt_summary.get('avg', '')},")
        if "avg" in pt_summary and "avg" in jt_summary:
            f.write(f"{jt_summary['avg'] - pt_summary['avg']:.6f}")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Plot PyTorch/Jittor RAPF metric comparison figures.")
    parser.add_argument(
        "--pytorch-metric",
        default="experiments/class/cifar100_10-10_without_exp/metric.json",
        help="PyTorch metric JSONL path.",
    )
    parser.add_argument(
        "--jittor-metric",
        default=None,
        help="Jittor metric JSONL path. If omitted, use the newest metric_jittor*.json file.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/class/cifar100_10-10_without_exp/figures",
        help="Directory for generated figures and summary CSV.",
    )
    args = parser.parse_args()

    if args.jittor_metric is None:
        metric_dir = os.path.dirname(args.pytorch_metric)
        candidates = [
            path
            for path in glob.glob(os.path.join(metric_dir, "metric_jittor*.json"))
            if not path.endswith(":Zone.Identifier")
        ]
        if not candidates:
            raise FileNotFoundError(f"No metric_jittor*.json found in {metric_dir}")
        args.jittor_metric = max(candidates, key=os.path.getmtime)

    ensure_dir(args.output_dir)
    pytorch_rows, pytorch_summary = load_metric_jsonl(args.pytorch_metric)
    jittor_rows, jittor_summary = load_metric_jsonl(args.jittor_metric)

    save_line_plot(
        pytorch_rows,
        jittor_rows,
        "acc",
        "Accuracy (%)",
        "Incremental Accuracy: PyTorch vs Jittor",
        os.path.join(args.output_dir, "acc_curve.png"),
    )
    save_line_plot(
        pytorch_rows,
        jittor_rows,
        "avg_acc",
        "Average Accuracy (%)",
        "Average Incremental Accuracy: PyTorch vs Jittor",
        os.path.join(args.output_dir, "avg_acc_curve.png"),
    )
    save_line_plot(
        pytorch_rows,
        jittor_rows,
        "forgetting",
        "Forgetting (%)",
        "Forgetting: PyTorch vs Jittor",
        os.path.join(args.output_dir, "forgetting_curve.png"),
    )
    save_line_plot(
        pytorch_rows,
        jittor_rows,
        "bwt",
        "Backward Transfer (%)",
        "Backward Transfer: PyTorch vs Jittor",
        os.path.join(args.output_dir, "bwt_curve.png"),
    )
    save_delta_plot(pytorch_rows, jittor_rows, "acc", os.path.join(args.output_dir, "acc_delta.png"))
    save_final_bar(pytorch_summary, jittor_summary, os.path.join(args.output_dir, "final_comparison.png"))
    save_heatmap(pytorch_rows, "PyTorch", os.path.join(args.output_dir, "acc_per_task_heatmap_pytorch.png"))
    save_heatmap(jittor_rows, "Jittor", os.path.join(args.output_dir, "acc_per_task_heatmap_jittor.png"))
    write_summary_table(
        pytorch_rows,
        jittor_rows,
        pytorch_summary,
        jittor_summary,
        os.path.join(args.output_dir, "metric_comparison.csv"),
    )
    print(f"Figures and comparison table saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
