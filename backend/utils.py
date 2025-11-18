# Copyright 2021-2024, Gavin E. Crooks
#
# This source code is licensed under the Apache-2.0 License
# found in the LICENSE file in the root directory of this source tree.


"""Lightweight utilities shared across API modules."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit


Array = NDArray[np.float64]
ArrayLike = ArrayLike  # Re-export for compatibility


def logexpit(a: ArrayLike) -> Array:
    """Stable computation of ``log(sigmoid(a))`` for real inputs."""
    values = np.asarray(a, dtype=np.float64)
    return np.where(
        values >= 0,
        -np.log1p(np.exp(-values)),
        values - np.log1p(np.exp(values)),
    )
