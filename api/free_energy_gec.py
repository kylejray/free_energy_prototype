# Copyright 2021-2024 Gavin E. Crooks
#
# This source code is licensed under the Apache License 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Estimates of free energy from work measurements.

All works and free energies are in multiples of kT. (i.e. nats)


Free Energy Estimates
---------------------
.. autofunction:: thermoflow.free_energy_bar

.. autofunction:: thermoflow.free_energy_bayesian

.. autofunction:: thermoflow.free_energy_logmeanexp

.. autofunction:: thermoflow.free_energy_logmeanexp_gaussian


Free Energy Estimates for symmetric protocols
--------------------------------------------
.. autofunction:: thermoflow.free_energy_symmetric_bar

.. autofunction:: thermoflow.free_energy_symmetric_bidirectional

.. autofunction:: thermoflow.free_energy_symmetric_nnznm"


"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root_scalar
from scipy.special import expit, logsumexp

from .utils import logexpit

__all__ = (
    "free_energy_bar",
    "free_energy_bayesian",
    "free_energy_logmeanexp",
    "free_energy_logmeanexp_gaussian",
    "free_energy_symmetric_bar",
    "free_energy_symmetric_bidirectional",
    "free_energy_symmetric_nnznm",
)


FloatArray = NDArray[np.float64]


def _as_float_array(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def free_energy_logmeanexp(work_f: ArrayLike) -> float:
    """
    Calculate the free energy difference from a series of work measurements
    using the Jarzynski equality.

    (Note that we can't calculate an accurate error bounds for Jarzynski estimate
    using measurements from only a single protocol directions.)

    Args:
        work_f: Array of shape [N]
    Returns:
        Delta free energy
    Ref:
        TODO
    """
    work_f_arr = _as_float_array(work_f)
    N_f = work_f_arr.size
    delta_free_energy = -(logsumexp(-work_f_arr) - np.log(N_f))

    return float(delta_free_energy)


def free_energy_logmeanexp_gaussian(work_f: ArrayLike) -> float:
    """
    Calculate the free energy difference from a series of work measurements
    under the assumption that the work distributions is Gaussian.

    Args:
        work_f: Array of shape [N]
    Returns:
        Delta free energy
    Ref:
        TODO
    """
    work_f_arr = _as_float_array(work_f)
    delta_free_energy = np.average(work_f_arr) - 0.5 * np.var(work_f_arr)
    return float(delta_free_energy)


def free_energy_bar(
    work_f: ArrayLike,
    work_r: ArrayLike,
    weights_f: Optional[ArrayLike] = None,
    weights_r: Optional[ArrayLike] = None,
    uncertainty_method: str = "BAR",
) -> Tuple[float, float]:
    """
    Estimate a free energy difference using the Bennett Acceptance Ratio method (BAR).

    Three methods for estimating the error are provided. The original 'BAR' method [1],
    which underestimates the error if the posterior is not Gaussian (this happens
    when there is little overlap between the forward and negative reverse work
    distributions and therefore the uncertainties are large); the 'MBAR' [3] approach which
    is optimal in the large sample limit, but overestimates the error when the
    posteriors are not Gaussian, and 'logistic', which is the MBAR error,
    but with a correction for non-overlapping work distributions.

    Args:
        work_f: Measurements of work from forward protocol.
        work_r: Measurements of work from reverse protocol.
        weights_f:  Optional weights for forward works
        weights_r:  Optional weights for reverse works
        uncertainty_method: Method to calculate errors ("BAR" [1], "MBAR"[3], or "Logistic"[4])

    Returns:
        Estimated free energy difference, and the estimated error
    Refs:
        ???, ???, ???, ???

    """
    W_f = _as_float_array(work_f)
    W_r = _as_float_array(work_r)

    weights_f_arr = _as_float_array(weights_f) if weights_f is not None else np.ones_like(W_f)
    weights_r_arr = _as_float_array(weights_r) if weights_r is not None else np.ones_like(W_r)

    N_f = float(np.sum(weights_f_arr))
    N_r = float(np.sum(weights_r_arr))
    if N_f <= 0 or N_r <= 0:
        raise ValueError("Weights must sum to positive values for both forward and reverse works.")

    M = np.log(N_f / N_r)

    lower = float(np.min([np.min(W_f), np.min(-W_r)]))
    upper = float(np.max([np.max(W_f), np.max(-W_r)]))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Work arrays must contain finite values.")
    if upper <= lower:
        upper = lower + 1e-6

    def _bar(delta_free_energy: float) -> float:
        diss_f = W_f - delta_free_energy + M
        diss_r = W_r + delta_free_energy - M

        f = np.log(np.sum(weights_f_arr * expit(-diss_f)))
        r = np.log(np.sum(weights_r_arr * expit(-diss_r)))
        return float(f - r)

    f_lower = _bar(lower)
    f_upper = _bar(upper)
    if np.sign(f_lower) == np.sign(f_upper):
        span = max(1.0, upper - lower)
        for _ in range(25):
            lower -= span
            upper += span
            f_lower = _bar(lower)
            f_upper = _bar(upper)
            if np.sign(f_lower) != np.sign(f_upper):
                break
            span *= 1.5
        else:
            raise RuntimeError("Unable to bracket BAR root with current data.")

    result = root_scalar(_bar, bracket=(lower, upper), method="bisect")
    if not result.converged:
        raise RuntimeError("BAR root finding failed to converge.")
    delta_free_energy = float(result.root)

    diss_f = W_f - delta_free_energy + M
    diss_r = W_r + delta_free_energy - M

    slogF = np.sum(weights_f_arr * expit(-diss_f))
    slogR = np.sum(weights_r_arr * expit(-diss_r))

    slogF2 = np.sum(weights_f_arr * expit(-diss_f) ** 2)
    slogR2 = np.sum(weights_r_arr * expit(-diss_r) ** 2)

    nratio = (N_f + N_r) / (N_f * N_r)

    if uncertainty_method == "BAR":
        err = np.sqrt((slogF2 / slogF**2) + (slogR2 / slogR**2) - nratio)
    elif uncertainty_method == "MBAR":
        err = np.sqrt(1.0 / (slogF - slogF2 + slogR - slogR2) - nratio)
    elif uncertainty_method == "Logistic":
        mbar_err = np.sqrt(1.0 / (slogF - slogF2 + slogR - slogR2) - nratio)
        min_hysteresis = float(np.min(W_f) + np.min(W_r))
        logistic_err = np.sqrt((min_hysteresis**2 + 4 * np.pi**2) / 12)
        err = min(logistic_err, mbar_err)
    else:
        raise ValueError("Unknown uncertainty estimation method")

    return delta_free_energy, float(err)


def free_energy_bayesian(work_f: ArrayLike, work_r: ArrayLike) -> Tuple[float, float]:
    """Bayesian free energy estimate

    Args:
        work_f: Measurements of work from forward protocol.
        work_r: Measurements of work from reverse protocol.

    Returns:
        Posterior mean estimate of the free energy difference, and the estimated error
    """
    df, prob = free_energy_posterior(work_f, work_r)

    delta_free_energy = float(np.sum(df * prob))
    err = float(np.sqrt(np.sum(df * df * prob) - delta_free_energy**2))

    return delta_free_energy, err


def free_energy_posterior(work_f: ArrayLike, work_r: ArrayLike) -> Tuple[FloatArray, FloatArray]:
    """The Bayesian free energy posterior distribution.

    Args:
        work_f: Measurements of work from forward protocol.
        work_r: Measurements of work from reverse protocol.
    Returns:
        energy and probability, pair of arrays of shapes [N]
    """

    w_f = _as_float_array(work_f)
    w_r = _as_float_array(work_r)

    fe, err = free_energy_bar(work_f, work_r, uncertainty_method="Logistic")
    lower = fe - 4 * err
    upper = fe + 4 * err

    x = np.linspace(lower, upper, 100, dtype=np.float64)

    N_f = w_f.size
    N_r = w_r.size
    M = np.log(N_f / N_r)

    def compute_log_prob(fe_value: float) -> float:
        diss_f = w_f - fe_value + M
        diss_r = w_r + fe_value - M
        return float(np.sum(logexpit(diss_f)) + np.sum(logexpit(diss_r)))

    log_prob = np.array([compute_log_prob(val) for val in x], dtype=np.float64)

    log_prob -= np.max(log_prob)
    prob = np.exp(log_prob)
    total = np.sum(prob)
    if total == 0.0:
        raise RuntimeError("Posterior probabilities underflowed to zero.")
    prob /= total

    return x, prob


def free_energy_symmetric_bar(
    work_ab: ArrayLike,
    work_bc: ArrayLike,
    uncertainty_method: str = "BAR",
) -> Tuple[float, float]:
    """BAR for symmetric periodic protocols.

    Args:
        work_ab: Measurements of work from first half of protocol.
        work_bc: Measurements of work from mirror image second half of protocol.
        uncertainty_method: Method to calculate errors (BAR, MBAR, or Logistic)

    Returns:
        Estimated free energy difference to the middle point of the protocol, and
        an estimated error
    """
    work_ab_arr = _as_float_array(work_ab)
    work_bc_arr = _as_float_array(work_bc)

    weights_r = np.exp(-work_ab_arr - free_energy_logmeanexp(work_ab_arr))
    return free_energy_bar(work_ab_arr, work_bc_arr, None, weights_r, uncertainty_method)


def free_energy_symmetric_nnznm(work_ab: ArrayLike, work_bc: ArrayLike) -> float:
    """Free energy estimate for cyclic protocol.

    "Non equilibrium path-ensemble averages for symmetric protocols"
    Nguyen, Ngo, Zerba, Noskov, & Minh (2009), Eq 2

    Args:
        work_ab: Measurements of work from first half of protocol.
        work_bc: Measurements of work from mirror image second half of protocol.
    Returns:
        Estimate of the free energy
    """
    work_ab_arr = _as_float_array(work_ab)
    work_bc_arr = _as_float_array(work_bc)

    delta_fenegy = (
        -np.log(2)
        + free_energy_logmeanexp(-work_ab_arr)
        + np.log(1 + np.exp(-free_energy_logmeanexp(-work_ab_arr - work_bc_arr)))
    )

    return float(delta_fenegy)


def free_energy_symmetric_bidirectional(
    work_ab: ArrayLike, work_bc: ArrayLike
) -> float:
    """
    The bidirectional Minh-Chodera free energy estimate specialized to a symmetric
    protocol.

    Delta F = (2/N) sum (e^W_ab + e^-W_bc)^-1)

    Args:
        work_ab: Measurements of work from first half of protocol.
        work_bc: Measurements of work from mirror image second half of protocol.
    Returns:
        Estimate of the free energy
    """
    work_ab_arr = _as_float_array(work_ab)
    work_bc_arr = _as_float_array(work_bc)

    N = work_ab_arr.size

    return float(
        -(logsumexp(-work_ab_arr + logexpit(-work_ab_arr - work_bc_arr)) - np.log(N / 2))
    )
