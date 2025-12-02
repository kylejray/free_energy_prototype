from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import List, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

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
    z_score: float = 1.96,
    sampling_mode: Literal["constant_effort", "best_case_assumptions"] = "constant_effort",
) -> dict[str, object]:
    plt.rcParams.update({'font.size': 18})
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

    p_c, _ = integrate.quad(dist.pdf, ll, ul)
    r_c_rev, _ = integrate.quad(r_dist.pdf, -ul, -ll)

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
        axes_top[1].axvspan(ll, ul, color="tab:blue", alpha=0.3, label=r"$C$")
        axes_top[1].axvspan(-ul, -ll, color="tab:orange", alpha=0.3, label=r"$C^\dagger$")

        for i, axis in enumerate(axes_top):
            axis.axvline(F, color="k", alpha=0.75)
            if i == 0:
                axis.legend(loc="upper right")
            axis.set_xlabel("x")

        sampling_top_image = _figure_to_base64(fig_top)

        fig_bottom, axes_bottom = plt.subplots(1, 2, figsize=(15, 4.5), sharex=True)
        axes_bottom[0].set_title("PDFs of F and R(-x)")
        axes_bottom[0].plot(x_grid, [dist.pdf(val) for val in x_grid], label="F(x)")
        axes_bottom[0].plot(-x_grid, [r_dist.pdf(val) for val in x_grid], label="R(-x)", linestyle="--", zorder=100)

        axes_bottom[1].set_title("PDFs of F and R(-x) (log scale)")
        axes_bottom[1].plot(x_grid, [dist.pdf(val) for val in x_grid], label="F(x)")
        axes_bottom[1].plot(-x_grid, [r_dist.pdf(val) for val in x_grid], label="R(-x)", linestyle="--", zorder=100)
        axes_bottom[1].set_yscale("log")

        for i, axis in enumerate(axes_bottom):
            axis.axvspan(ll, ul, color="tab:grey", alpha=0.3)
            axis.axvline(F, color="k", alpha=0.75)
            if i == 0:
                axis.legend(loc="upper right")
            axis.set_xlabel("x")

        sampling_bottom_image = _figure_to_base64(fig_bottom)

    bar_methods = ("BAR", "MBAR", "Logistic")
    BAR_TC: list[list[float]] = []
    BAR_TC_CORR: list[list[float]] = []
    BAR_FULL: list[list[float]] = []
    JAR_TC: list[list[float]] = []
    JAR_TC_CORR: list[list[float]] = []
    JAR_FULL: list[list[float]] = []

    BAR_TC_COUNTS: list[tuple[int, int]] = []
    BAR_FULL_COUNTS: list[tuple[int, int]] = []
    JAR_TC_COUNTS: list[int] = []
    JAR_FULL_COUNTS: list[int] = []

    standard_bar_results: dict[str, list[list[float]]] = {method: [] for method in bar_methods}

    failed_trials = 0
    failure_reasons = {
        "timeout_forward": 0,
        "timeout_reverse": 0,
        "empty_forward": 0,
        "empty_reverse": 0,
        "insufficient_forward": 0,
        "insufficient_reverse": 0,
        "empty_class_forward": 0,
        "empty_class_reverse": 0,
        "sampling_error": 0
    }

    if include_free_energy or include_standard:
        # Constant effort: Total samples = subset_size
        # JAR: subset_size forward samples
        # BAR: subset_size/2 forward + subset_size/2 reverse samples
        n_total = subset_size
        n_half = max(1, n_total // 2)

        for _ in range(trials):
            if sampling_mode == "best_case_assumptions":
                # Best Case Assumptions Mode
                # JAR_TC: Needs exactly subset_size samples in class
                # BAR_TC: Needs exactly n_half samples in class (F & R)
                
                # --- Forward Sampling ---
                f_all = []
                f_in_class_count = 0
                batch_size = max(100, subset_size * 5)
                max_samples = 1_000_000
                t_start = time.time()
                
                while f_in_class_count < subset_size and len(f_all) * batch_size < max_samples:
                    if time.time() - t_start > 2.0:  # 2.0s timeout per leg
                        break
                    batch = hdist.rejection_sample(batch_size)
                    mask = (batch > ll) & (batch < ul)
                    f_in_class_count += np.sum(mask)
                    f_all.append(batch)
                
                if not f_all:
                    failed_trials += 1
                    failure_reasons["empty_forward"] += 1
                    break
                
                if f_in_class_count < subset_size and time.time() - t_start > 10.0:
                     failed_trials += 1
                     failure_reasons["timeout_forward"] += 1
                     break

                forward_draw_full = np.concatenate(f_all)
                
                # Truncate forward_draw_full to exactly the point where we got the Nth class sample
                mask_full = (forward_draw_full > ll) & (forward_draw_full < ul)
                valid_indices = np.where(mask_full)[0]
                
                if len(valid_indices) < subset_size:
                    failed_trials += 1
                    failure_reasons["insufficient_forward"] += 1
                    break # Failed to get enough samples
                
                cutoff_jar = valid_indices[subset_size - 1]
                forward_draw_full = forward_draw_full[:cutoff_jar+1]
                
                # For BAR, we need n_half samples. Since subset_size >= n_half, we can reuse forward_draw_full.
                # We truncate it further for BAR usage.
                cutoff_bar = valid_indices[n_half - 1]
                forward_subset_bar = forward_draw_full[:cutoff_bar+1]
                
                # --- Reverse Sampling ---
                r_all = []
                r_in_class_count = 0
                t_start = time.time()
                
                while r_in_class_count < n_half and len(r_all) * batch_size < max_samples:
                    if time.time() - t_start > 2.0:  # 2.0s timeout per leg
                        break
                    batch = r_hdist.rejection_sample(batch_size)
                    mask = (batch < -ll) & (batch > -ul)
                    r_in_class_count += np.sum(mask)
                    r_all.append(batch)
                
                if not r_all:
                    failed_trials += 1
                    failure_reasons["empty_reverse"] += 1
                    break

                if r_in_class_count < n_half and time.time() - t_start > 2.0:
                     failed_trials += 1
                     failure_reasons["timeout_reverse"] += 1
                     break

                reverse_draw_subset = np.concatenate(r_all)
                
                mask_r = (reverse_draw_subset < -ll) & (reverse_draw_subset > -ul)
                valid_indices_r = np.where(mask_r)[0]
                
                if len(valid_indices_r) < n_half:
                    failed_trials += 1
                    failure_reasons["insufficient_reverse"] += 1
                    break
                
                cutoff_r = valid_indices_r[n_half - 1]
                reverse_subset_bar = reverse_draw_subset[:cutoff_r+1]

            else:
                # Constant Effort Mode
                forward_draw_full = hdist.rejection_sample(n_total)
                reverse_draw_subset = r_hdist.rejection_sample(n_half)

                if (
                    forward_draw_full.size < n_total
                    or reverse_draw_subset.size < n_half
                ):
                    failed_trials += 1
                    failure_reasons["sampling_error"] += 1
                    continue

                # BAR subsets (N/2 each)
                forward_subset_bar = forward_draw_full[:n_half]
                reverse_subset_bar = reverse_draw_subset[:n_half]

            if include_standard:
                for method in bar_methods:
                    try:
                        estimate, error = free_energy_bar(
                            forward_subset_bar,
                            reverse_subset_bar,
                            uncertainty_method=method,
                        )
                    except Exception:  # pragma: no cover - defensive guard against numerical issues
                        continue

                    estimate_f = float(np.asarray(estimate))
                    error_f = float(np.asarray(error))
                    if np.isfinite(estimate_f) and np.isfinite(error_f):
                        standard_bar_results[method].append([estimate_f, error_f])

            if include_free_energy:
                # Prepare inputs for Full estimators
                if sampling_mode == "best_case_assumptions":
                    jar_full_input = forward_draw_full[:subset_size]
                    bar_full_f_input = forward_subset_bar[:n_half]
                    bar_full_r_input = reverse_subset_bar[:n_half]
                else:
                    jar_full_input = forward_draw_full
                    bar_full_f_input = forward_subset_bar
                    bar_full_r_input = reverse_subset_bar

                # JAR uses full forward dataset (N)
                f_class_jar = (forward_draw_full > ll) & (forward_draw_full < ul)
                
                # BAR uses split datasets (N/2 each)
                f_class_bar = (forward_subset_bar > ll) & (forward_subset_bar < ul)
                r_class_bar = (reverse_subset_bar < -ll) & (reverse_subset_bar > -ul)

                if np.sum(f_class_jar) == 0:
                    failed_trials += 1
                    failure_reasons["empty_class_forward"] += 1
                    continue
                if np.sum(f_class_bar) == 0:
                    failed_trials += 1
                    failure_reasons["empty_class_forward"] += 1
                    continue
                if np.sum(r_class_bar) == 0:
                    failed_trials += 1
                    failure_reasons["empty_class_reverse"] += 1
                    continue

                # Jarzynski Estimates (N samples)
                jar_full = class_meta_f(jar_full_input, sample_error=True)
                jar_tc = class_meta_f(forward_draw_full[f_class_jar], sample_error=True)
                
                # Correction term (using BAR split data for consistency)
                if sampling_mode == "best_case_assumptions":
                    tcft_mean = -np.log(p_c / r_c_rev)
                    tcft_var = 0.0
                else:
                    tcft_mean, tcft_var = tcft_correction(f_class_bar, r_class_bar)
                
                jar_tc_corr = [jar_tc[0] + tcft_mean, np.sqrt(jar_tc[1] ** 2 + tcft_var)]

                # BAR Estimates (N/2 + N/2 samples)
                bar_full = free_energy_bar(bar_full_f_input, bar_full_r_input)
                bar_tc = free_energy_bar(
                    forward_subset_bar[f_class_bar],
                    reverse_subset_bar[r_class_bar],
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

                JAR_FULL_COUNTS.append(len(jar_full_input))
                JAR_TC_COUNTS.append(np.sum(f_class_jar))
                BAR_FULL_COUNTS.append((len(bar_full_f_input), len(bar_full_r_input)))
                BAR_TC_COUNTS.append((np.sum(f_class_bar), np.sum(r_class_bar)))

    if include_free_energy:
        avg_jar_tc = np.mean(JAR_TC_COUNTS) if JAR_TC_COUNTS else 0
        avg_jar_full = np.mean(JAR_FULL_COUNTS) if JAR_FULL_COUNTS else 0
        avg_bar_tc = np.mean(BAR_TC_COUNTS, axis=0) if BAR_TC_COUNTS else (0, 0)
        avg_bar_full = np.mean(BAR_FULL_COUNTS, axis=0) if BAR_FULL_COUNTS else (0, 0)

        datas = [BAR_TC, BAR_TC_CORR, BAR_FULL, JAR_TC, JAR_TC_CORR, JAR_FULL]
        labels = [
            r"$\Delta F_{\text{BAR}}(C)$",
            r"$\Delta F_{\text{BAR}}(C) - \ln(P/R)$",
            r"$\Delta F_{\text{BAR}}$",
            r"$\Delta F_{\text{JAR}}(C)$",
            r"$\Delta F_{\text{JAR}}(C) - \ln(P/R)$",
            r"$\Delta F_{\text{JAR}}$",
        ]
        
        titles = [
            rf"$\langle N_F \rangle \approx {int(round(avg_bar_tc[0]))}, \langle N_R \rangle \approx {int(round(avg_bar_tc[1]))}$",
            rf"$\langle N_F \rangle \approx {int(round(avg_bar_tc[0]))}, \langle N_R \rangle \approx {int(round(avg_bar_tc[1]))}$",
            f"N_F={avg_bar_full[0]:.1f}, N_R={avg_bar_full[1]:.1f}",
            rf"$\langle N \rangle \approx {int(round(avg_jar_tc))}$",
            rf"$\langle N \rangle \approx {int(round(avg_jar_tc))}$",
            f"N={avg_jar_full:.1f}",
        ]

        old_ylim_lower, old_ylim_upper = _free_energy_limits(datas, F)

        fig_var_top, axes_var_top = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
        fig_var_bottom, axes_var_bottom = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

        for i, (data, label, axis) in enumerate(zip(datas[:3], labels[:3], axes_var_top)):
            if not data:
                axis.set_visible(False)
                continue
            variance_plot(data, ax=axis, parameter=F, z_score=z_score, show_legend=(i == 0))
            current_title = axis.get_title()
            axis.set_title(f"{titles[i]}\n{current_title}")
            axis.set_ylabel(label)
            axis.set_xlabel("trial number")
            axis.set_ylim(old_ylim_lower, old_ylim_upper)

        for i, (data, label, axis) in enumerate(zip(datas[3:], labels[3:], axes_var_bottom)):
            if not data:
                axis.set_visible(False)
                continue
            variance_plot(data, ax=axis, parameter=F, z_score=z_score, show_legend=(i == 0))
            current_title = axis.get_title()
            axis.set_title(f"{titles[i+3]}\n{current_title}")
            axis.set_ylabel(label)
            axis.set_xlabel("trial number")
            axis.set_ylim(old_ylim_lower, old_ylim_upper)

        free_energy_top_image = _figure_to_base64(fig_var_top)
        free_energy_bottom_image = _figure_to_base64(fig_var_bottom)

    if include_standard:
        standard_datasets = list(standard_bar_results.values())
        if any(len(data) > 0 for data in standard_datasets):
            standard_ylim_lower, standard_ylim_upper = _free_energy_limits(standard_datasets, F)

            fig_standard, axes_standard = plt.subplots(
                1,
                len(bar_methods),
                figsize=(18, 5),
                sharex=True,
                sharey=True,
            )

            axes_iter = np.atleast_1d(axes_standard).flatten()
            for i, (axis, method) in enumerate(zip(axes_iter, bar_methods)):
                data = standard_bar_results[method]
                if not data:
                    axis.set_visible(False)
                    continue
                variance_plot(data, ax=axis, parameter=F, z_score=z_score, show_legend=(i == 0))
                axis.set_ylabel(rf"$\Delta F_{{\text{{BAR}}}}$ ({method} error)")
                axis.set_xlabel("trial number")
                axis.set_ylim(standard_ylim_lower, standard_ylim_upper)

            free_energy_standard_image = _figure_to_base64(fig_standard)

    metadata = {
        "F": F,
        "sample_size": float(sample_size),
        "trials": float(trials),
        "subset_size": float(subset_size),
        "p_c": p_c,
        "r_c_rev": r_c_rev,
        "failed_trials": failed_trials,
        "failure_reasons": failure_reasons,
    }

    return {
        "sampling_top_plot": sampling_top_image,
        "sampling_bottom_plot": sampling_bottom_image,
        "free_energy_top_plot": free_energy_top_image,
        "free_energy_bottom_plot": free_energy_bottom_image,
        "free_energy_standard_plot": free_energy_standard_image,
        "metadata": metadata,
    }
