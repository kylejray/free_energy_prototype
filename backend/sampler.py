from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from random import gauss


@dataclass(slots=True)
class Histogram:
    edges: list[float]
    counts: list[int]


@dataclass(slots=True)
class SampleStats:
    sample_mean: float
    sample_variance: float
    requested_mean: float
    requested_variance: float


@dataclass(slots=True)
class SampleResult:
    samples: list[float]
    histogram: Histogram
    stats: SampleStats


def _resolve_bin_count(size: int, bins: int | None) -> int:
    if bins is not None:
        return max(1, min(80, bins))
    heuristically = max(15, min(50, size // 10))
    return max(1, heuristically)


def normal_samples(mean: float, variance: float, size: int, bins: int | None = None) -> SampleResult:
    if size < 1:
        raise ValueError("Size must be at least 1.")
    if variance < 0:
        raise ValueError("Variance must be non-negative.")

    deviation = sqrt(variance)
    samples = [gauss(mean, deviation) for _ in range(size)]

    minimum = min(samples)
    maximum = max(samples)

    if minimum == maximum:
        edges = [minimum - 0.5, maximum + 0.5]
        counts = [size]
    else:
        bin_count = _resolve_bin_count(size, bins)
        span = maximum - minimum
        bin_width = span / bin_count if span else 1.0
        edges = [minimum + bin_width * index for index in range(bin_count + 1)]
        counts = [0 for _ in range(bin_count)]

        for value in samples:
            index = int((value - minimum) / bin_width)
            if index >= bin_count:
                index = bin_count - 1
            counts[index] += 1

    sample_mean = sum(samples) / size
    sample_variance = (
        sum((value - sample_mean) ** 2 for value in samples) / (size - 1)
        if size > 1
        else 0.0
    )

    return SampleResult(
        samples=samples,
        histogram=Histogram(edges=edges, counts=counts),
        stats=SampleStats(
            sample_mean=sample_mean,
            sample_variance=sample_variance,
            requested_mean=mean,
            requested_variance=variance,
        ),
    )
