from __future__ import annotations

from typing import Iterable, List, Optional

import tensorflow as tf
import tf_keras as keras


class ExponentialMovingAverage:
    """Shadow weight tracker for EMA."""

    def __init__(self, model: keras.Model, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: List[tf.Variable] = []
        self.backup: Optional[List[tf.Variable]] = None
        self._initialize()

    def _initialize(self):
        # Ensure variables exist
        _ = [w.numpy() for w in self.model.weights]
        self.shadow = [tf.Variable(w, trainable=False, dtype=w.dtype) for w in self.model.weights]

    def update(self):
        for s, w in zip(self.shadow, self.model.weights):
            s.assign(self.decay * s + (1.0 - self.decay) * w)

    def apply_shadow(self):
        self.backup = [tf.Variable(w, trainable=False, dtype=w.dtype) for w in self.model.weights]
        for w, s in zip(self.model.weights, self.shadow):
            w.assign(s)

    def restore(self):
        if self.backup is None:
            return
        for w, b in zip(self.model.weights, self.backup):
            w.assign(b)
        self.backup = None
