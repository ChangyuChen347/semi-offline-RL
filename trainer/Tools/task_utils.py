import torch
from dataclasses import dataclass
from typing import List
import numpy
from torch.utils import data
from transformers.trainer_callback import TrainerState
from trainer.Tools.plot_utils import plot_empirical_stats
import os


@dataclass
class TaskInfo:
    tasks: List[str] = None

    def __post_init__(self):
        self.task_ids = list(range(len(self.tasks)))
        self.task_to_id_map = dict(zip(self.tasks, self.task_ids))

    def convert_task_to_id(self, task, return_tensor):
        task_ids = self.task_to_id_map[task]
        if return_tensor == 'np':
            return np.array(task_ids)
        elif return_tensor == "torch.tensor":
            return torch.tensor(task_ids)

        return task_ids


@dataclass
class RobustTrainerState(TrainerState):
    plot_dir: str = None

    def __post_init__(self):
        super().__post_init__()
        self.loss_stats = []
        self.distribution_stats = []
        os.makedirs(self.plot_dir, exist_ok=True)

    def update_stats(self):
        self.loss_stats.append(self.task_avg_loss.tolist())
        self.distribution_stats.append(self.task_dist.tolist())

    def update_plots(self, task_labels):
        plot_empirical_stats(self.loss_stats, task_labels, "loss", self.plot_dir)
        plot_empirical_stats(self.distribution_stats, task_labels, "distribution", self.plot_dir)
