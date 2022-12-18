import logging
import math
import time

from typing import Dict
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class RunningAverage:
    def __init__(self):
        self._means = defaultdict(float)
        self._count = defaultdict(int)
        self._cache = defaultdict(float)

    def add(self, tag, scalar_value):
        if math.isnan(scalar_value):
            logger.warning(f'Value {tag!r} is NaN')
        else:
            self._count[tag] += 1
            self._means[tag] += (scalar_value - self._means[scalar_value]) / self._count[scalar_value]

    def mean(self, tag):
        if self._count[tag] > 0:
            self._cache[tag] = self._means[tag]
            self._reset(tag)
        return self._cache[tag]

    def add_dict(self, value_dict):
        for name, value in value_dict.items():
            self.add(name, value)

    def mean_dict(self) -> Dict[str, float]:
        return {name: self.mean(name) for name in self._count.keys()}

    def _reset(self, name: str) -> None:
        self._count[name] = self._count.default_factory()
        self._means[name] = self._means.default_factory()

    def __str__(self):
        means = self.mean_dict()
        return "\n" + "\n".join(f"{key}: {val}" for key, val in means.items())


class TensorboardWriter(SummaryWriter):
    def __init__(self, log_dir, default_window=100):
        super().__init__(log_dir=log_dir)
        self._default_window = default_window
        self._running_average = RunningAverage()
        self._current_step = 0

    def add_scalar(self, tag, scalar_value, global_step=None):
        if torch.is_tensor(scalar_value):
            scalar_value = scalar_value.item()

        super().add_scalar(tag, scalar_value, global_step)

    def _dump_windowed_scalar(self, tag, global_step):
        scalar_value = self._running_average.mean(tag)
        self.add_scalar(tag, scalar_value, global_step)

    def add_windowed_scalar(self, tag, scalar_value, global_step=None):
        if global_step is None:
            self._current_step += 1
            global_step = self._current_step

        self._current_step = global_step

        if torch.is_tensor(scalar_value):
            scalar_value = scalar_value.item()

        self._running_average.add(tag, scalar_value)
        if global_step % self._default_window == 0:
            self._dump_windowed_scalar(tag, global_step)


class TensorboardTimer:
    def __init__(self, scope_name, writer):
        self.scope_name = scope_name
        self._writer = writer

    def __enter__(self):
        self.start = time.time() * 1000
        return self

    def __exit__(self, *args):
        self.end = time.time() * 1000
        interval = int(self.end - self.start)

        self._writer.add_windowed_scalar(
            tag=self.scope_name,
            scalar_value=interval
        )
