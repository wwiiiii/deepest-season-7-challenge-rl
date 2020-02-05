import random
from collections import deque

import numpy as np

from utils import DataSchema


class Replay:
    def __init__(self, max_size: int, schema: DataSchema):
        self._size = 0
        self._next_idx = 0
        self._max_size = max_size
        self._storage = [deque([], maxlen=max_size) for _ in schema.names]
        self.schema = schema

    def push(self, *data):
        assert len(data) == len(self._storage)
        for v, arr in zip(data, self._storage):
            arr.append(v)

    def sample(self, n):
        idxes = random.choices(range(len(self._storage[0])), k=n)

        ret = [
            np.array([arr[i] for i in idxes]).reshape((n, *shape)).astype(dtype)
            for arr, shape, dtype
            in zip(self._storage, self.schema.shapes, self.schema.dtypes)
        ]

        return ret
