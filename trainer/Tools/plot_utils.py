import matplotlib.pyplot as plt
from typing import List


def plot_empirical_stats(task_stats: List[List], task_labels: List, stats_name: str, plot_dir):
    plt.figure()
    _task_stats = list(zip(*task_stats))
    for task_id, task_label in enumerate(task_labels):
        plt.plot(_task_stats[task_id], label=task_label)

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(stats_name)
    plt.savefig(f"{plot_dir}/{stats_name}_across_tasks.png")