import random

import numpy as np
import torch


class DataSchema:
    def __init__(self, names, shapes, dtypes):
        self.names = names
        self.shapes = shapes
        self.dtypes = dtypes


class LinearEpsilonScheduler:
    def __init__(
            self,
            initial_eps=1.0,
            final_eps=0.01,
            initial_exploration_frame=100000,
            max_exploration_frame=300000,
    ):
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.init_frame = initial_exploration_frame
        self.max_frame = max_exploration_frame

    def get_epsilon(self, frame):
        if frame < self.init_frame:
            return self.initial_eps
        elif frame > self.max_frame:
            return self.final_eps
        else:
            progress = (frame - self.init_frame) / (self.max_frame - self.init_frame)
            return self.initial_eps + (self.final_eps - self.initial_eps) * progress


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
