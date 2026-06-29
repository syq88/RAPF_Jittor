import numpy as np


class IncrementalMetricLogger:
    def __init__(self):
        self.task_results = []
        self.current = []

    def add(self, preds, targets, task_ids):
        preds = np.asarray(preds)
        targets = np.asarray(targets)
        task_ids = np.asarray(task_ids)
        self.current.append((preds, targets, task_ids))

    def _current_per_task(self):
        if not self.current:
            return []
        preds = np.concatenate([x[0] for x in self.current])
        targets = np.concatenate([x[1] for x in self.current])
        task_ids = np.concatenate([x[2] for x in self.current])
        per_task = []
        for task_id in sorted(np.unique(task_ids).tolist()):
            mask = task_ids == task_id
            per_task.append(float((preds[mask] == targets[mask]).mean()))
        return per_task

    def _results_with_current(self):
        # 写 metric.json 时是在 end_task 之前读取指标，因此这里要把当前 task 的评测结果一起算进去。
        current = self._current_per_task()
        if current:
            return self.task_results + [current]
        return self.task_results

    def end_task(self):
        self.task_results.append(self._current_per_task())
        self.current = []

    @property
    def accuracy(self):
        if self.current:
            preds = np.concatenate([x[0] for x in self.current])
            targets = np.concatenate([x[1] for x in self.current])
            return float((preds == targets).mean())
        if not self.task_results:
            return 0.0
        return float(np.mean(self.task_results[-1]))

    @property
    def accuracy_per_task(self):
        results = self._results_with_current()
        return results[-1] if results else []

    @property
    def average_incremental_accuracy(self):
        results = self._results_with_current()
        if not results:
            return 0.0
        return float(np.mean([np.mean(x) for x in results if x]))

    @property
    def forgetting(self):
        results = self._results_with_current()
        if len(results) <= 1:
            return 0.0
        last = results[-1]
        forgetting = []
        for task_id in range(len(last) - 1):
            best_before = max(result[task_id] for result in results[:-1] if task_id < len(result))
            forgetting.append(best_before - last[task_id])
        return float(np.mean(forgetting)) if forgetting else 0.0

    @property
    def backward_transfer(self):
        return -self.forgetting

    @property
    def forward_transfer(self):
        return 0.0
