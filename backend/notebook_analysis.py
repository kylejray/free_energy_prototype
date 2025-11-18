from __future__ import annotations

import base64
from io import BytesIO
from typing import List, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bimodal_dist import (
    HistDist,
    dist_from_piecewise_list,
    rdist_from_piecesise_dist,
)
from .free_energy import class_meta_f, tcft_correction, variance_plot
from .free_energy_gec import free_energy_bar


AnalysisSection = Literal["sampling", "free_energy", "standard", "all"]


def _figure_to_base64(fig: plt.Figure) -> str:
    buffer = BytesIO()
    try:
        fig.tight_layout()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("ascii")
    finally:
        plt.close(fig)
        buffer.close()


def _free_energy_limits(datasets: list[list[list[float]]], center: float) -> tuple[float, float]:
    """Compute a zoomed-in y-window around the target free energy."""
    points = [estimate for data in datasets for estimate, _ in data]
    if not points:
        span = 1.0
    else:
        deviations = np.abs(np.asarray(points) - center)
        span = float(np.max(deviations) + 0.25) if deviations.size else 1.0
        span = max(span, 0.75)
    return center - span, center + span


def run_notebook_analysis(
    xp: List[float],
    fp: List[float],
    ll: float,
    ul: float,
    *,
    section: AnalysisSection = "all",
    histogram_resolution: int = 5_000,
    sample_size: int = 25_000,
    trials: int = 50,
    subset_size: int = 15,
) -> dict[str, object]:
    if len(xp) != len(fp):
        raise ValueError("`xp` and `fp` must have the same length.")
    if len(xp) < 4:
        raise ValueError("`xp` and `fp` should contain at least four points to define a distribution.")
    if ll >= ul:
        raise ValueError("Lower limit `LL` must be smaller than upper limit `UL`.")
    if sample_size <= 0:
        raise ValueError("`sample_size` must be a positive integer.")
    if trials <= 0:
        raise ValueError("`trials` must be a positive integer.")

    sample_size = int(sample_size)
    trials = int(trials)
    subset_size = int(subset_size)
    if subset_size <= 0:
        raise ValueError("`subset_size` must be a positive integer.")

    section_normalized = section.lower()
    if section_normalized not in {"sampling", "free_energy", "standard", "all"}:
        raise ValueError("section must be 'sampling', 'free_energy', 'standard', or 'all'.")

    include_sampling = section_normalized in {"sampling", "all"}
    include_free_energy = section_normalized in {"free_energy", "all"}
    include_standard = section_normalized in {"standard", "all"}

    xp_array = np.asarray(xp, dtype=float)
    fp_array = np.asarray(fp, dtype=float)

    dist = dist_from_piecewise_list(xp_array, fp_array)
    r_dist = rdist_from_piecesise_dist(dist)

    F = float(-np.log(dist.negexp_avg()))

    bins = np.linspace(dist.lims[0], dist.lims[-1], histogram_resolution)
    rbins = -bins[::-1]
    hdist = HistDist(dist.pdf, bins)
    r_hdist = HistDist(r_dist.pdf, rbins)

    sampling_top_image: str | None = None
    sampling_bottom_image: str | None = None
    free_energy_top_image: str | None = None
    free_energy_bottom_image: str | None = None
    free_energy_standard_image: str | None = None

    X = float(np.max(np.abs(xp_array)) + 1)
    x_grid = np.linspace(-X, X, 3_000)

    if include_sampling:
        forward_samples = hdist.rejection_sample(sample_size)
        reverse_samples = r_hdist.rejection_sample(sample_size)

        fig_top, axes_top = plt.subplots(1, 2, figsize=(15, 4.5), sharex=True)
        axes_top[0].set_title("Histogram of samples from F and R")
        axes_top[0].hist(forward_samples, bins=50, density=True, alpha=0.6, label="F samples")
        axes_top[0].hist(reverse_samples, bins=50, density=True, alpha=0.6, label="R samples")
        axes_top[0].plot(x_grid, [dist.pdf(val) for val in x_grid], label="F(x)")
        axes_top[0].plot(x_grid, [r_dist.pdf(val) for val in x_grid], label="R(x)")

        axes_top[1].set_title("PDFs of F and R")
        axes_top[1].plot(x_grid, [dist.pdf(val) for val in x_grid], label="F(x)", alpha=0.85)
        axes_top[1].plot(x_grid, [r_dist.pdf(val) for val in x_grid], label="R(x)", alpha=0.85)
        axes_top[1].axvspan(ll, ul, color="tab:blue", alpha=0.3, label="C")
        axes_top[1].axvspan(-ul, -ll, color="tab:orange", alpha=0.3, label="C^R")

        for axis in axes_top:
            axis.axvline(F, color="k", alpha=0.75)
            axis.legend(loc="upper right")

        sampling_top_image = _figure_to_base64(fig_top)

        fig_bottom, axes_bottom = plt.subplots(1, 2, figsize=(15, 4.5), sharex=True)
        axes_bottom[0].set_title("PDFs of F and R(-x)")
        axes_bottom[0].plot(x_grid, [dist.pdf(val) for val in x_grid], label="F(x)")
        axes_bottom[0].plot(-x_grid, [r_dist.pdf(val) for val in x_grid], label="R(-x)", linestyle="--", zorder=100)

        axes_bottom[1].set_title("PDFs of F and R(-x) (log scale)")
        axes_bottom[1].plot(x_grid, [dist.pdf(val) for val in x_grid], label="F(x)")
        axes_bottom[1].plot(-x_grid, [r_dist.pdf(val) for val in x_grid], label="R(-x)", linestyle="--", zorder=100)
        axes_bottom[1].set_yscale("log")

        for axis in axes_bottom:
            axis.axvspan(ll, ul, color="tab:grey", alpha=0.3)
            axis.axvline(F, color="k", alpha=0.75)
            axis.legend(loc="upper right")

        sampling_bottom_image = _figure_to_base64(fig_bottom)

    bar_methods = ("BAR", "MBAR", "Logistic")
    BAR_TC: list[list[float]] = []
    BAR_TC_CORR: list[list[float]] = []
    BAR_FULL: list[list[float]] = []
    JAR_TC: list[list[float]] = []
    JAR_TC_CORR: list[list[float]] = []
    JAR_FULL: list[list[float]] = []

    standard_bar_results: dict[str, list[list[float]]] = {method: [] for method in bar_methods}

    if include_free_energy or include_standard:
        forward_sample_count = 2 * subset_size if include_free_energy else subset_size

        for _ in range(trials):
            forward_draw_full = hdist.rejection_sample(forward_sample_count)
            reverse_draw_subset = r_hdist.rejection_sample(subset_size)

            if (
                forward_draw_full.size < forward_sample_count
                or reverse_draw_subset.size < subset_size
            ):
                continue

            forward_subset = forward_draw_full[:subset_size]

            if include_standard:
                for method in bar_methods:
                    try:
                        estimate, error = free_energy_bar(
                            forward_subset,
                            reverse_draw_subset,
                            uncertainty_method=method,
                        )
                    except Exception:  # pragma: no cover - defensive guard against numerical issues
                        continue

                    estimate_f = float(np.asarray(estimate))
                    error_f = float(np.asarray(error))
                    if np.isfinite(estimate_f) and np.isfinite(error_f):
                        standard_bar_results[method].append([estimate_f, error_f])

            if include_free_energy:
                if forward_draw_full.size < 2 * subset_size:
                    continue

                f_class = (forward_subset > ll) & (forward_subset < ul)
                r_class = (reverse_draw_subset < -ll) & (reverse_draw_subset > -ul)

                if np.sum(f_class) == 0 or np.sum(r_class) == 0:
                    continue

                jar_full = class_meta_f(forward_draw_full[: 2 * subset_size])
                jar_tc = class_meta_f(forward_subset[f_class])
                tcft_mean, tcft_var = tcft_correction(f_class, r_class)
                jar_tc_corr = [jar_tc[0] + tcft_mean, np.sqrt(jar_tc[1] ** 2 + tcft_var)]

                bar_full = free_energy_bar(forward_subset, reverse_draw_subset)
                bar_tc = free_energy_bar(
                    forward_subset[f_class],
                    reverse_draw_subset[r_class],
                    uncertainty_method="MBAR",
                )
                bar_tc_corr = [
                    float(bar_tc[0] + tcft_mean),
                    float(np.sqrt(bar_tc[1] ** 2 + tcft_var)),
                ]

                BAR_TC.append([float(bar_tc[0]), float(bar_tc[1])])
                BAR_TC_CORR.append(bar_tc_corr)
                BAR_FULL.append([float(bar_full[0]), float(bar_full[1])])
                JAR_TC.append([float(jar_tc[0]), float(jar_tc[1])])
                JAR_TC_CORR.append([float(jar_tc_corr[0]), float(jar_tc_corr[1])])
                JAR_FULL.append([float(jar_full[0]), float(jar_full[1])])

    if include_free_energy:
        datas = [BAR_TC, BAR_TC_CORR, BAR_FULL, JAR_TC, JAR_TC_CORR, JAR_FULL]
        labels = [
            "BAR_TC",
            "BAR_TC - log(P/R)",
            "BAR Full",
            "JAR_TC",
            "JAR_TC - log(P/R)",
            "JAR Full",
        ]

        z_score = 1.64
        old_ylim_lower, old_ylim_upper = _free_energy_limits(datas, F)

        fig_var_top, axes_var_top = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
        fig_var_bottom, axes_var_bottom = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

        for (data, label, axis) in zip(datas[:3], labels[:3], axes_var_top):
            if not data:
                axis.set_visible(False)
                continue
            variance_plot(data, ax=axis, parameter=F, z_score=z_score)
            axis.set_ylabel(label)
            axis.set_ylim(old_ylim_lower, old_ylim_upper)

        for (data, label, axis) in zip(datas[3:], labels[3:], axes_var_bottom):
            if not data:
                axis.set_visible(False)
                continue
            variance_plot(data, ax=axis, parameter=F, z_score=z_score)
            axis.set_ylabel(label)
            axis.set_ylim(old_ylim_lower, old_ylim_upper)

        free_energy_top_image = _figure_to_base64(fig_var_top)
        free_energy_bottom_image = _figure_to_base64(fig_var_bottom)

    if include_standard:
        standard_datasets = list(standard_bar_results.values())
        if any(len(data) > 0 for data in standard_datasets):
            z_score = 1.64
            standard_ylim_lower, standard_ylim_upper = _free_energy_limits(standard_datasets, F)

            fig_standard, axes_standard = plt.subplots(
                1,
                len(bar_methods),
                figsize=(18, 5),
                sharex=True,
                sharey=True,
            )

            axes_iter = np.atleast_1d(axes_standard).flatten()
            for axis, method in zip(axes_iter, bar_methods):
                data = standard_bar_results[method]
                if not data:
                    axis.set_visible(False)
                    continue
                variance_plot(data, ax=axis, parameter=F, z_score=z_score)
                axis.set_ylabel(f"BAR ({method} error)")
                axis.set_ylim(standard_ylim_lower, standard_ylim_upper)

            free_energy_standard_image = _figure_to_base64(fig_standard)

    metadata = {
        "F": F,
        "sample_size": float(sample_size),
        "trials": float(trials),
        "subset_size": float(subset_size),
    }

    return {
        "sampling_top_plot": sampling_top_image,
        "sampling_bottom_plot": sampling_bottom_image,
        "free_energy_top_plot": free_energy_top_image,
        "free_energy_bottom_plot": free_energy_bottom_image,
        "free_energy_standard_plot": free_energy_standard_image,
        "metadata": metadata,
    }
