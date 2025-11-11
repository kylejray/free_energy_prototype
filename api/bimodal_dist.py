import numpy as np
import scipy as sp
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


def gauss(x, sig, mu):
    """Simple Gaussian pdf."""
    return np.exp(-((x - mu) ** 2) / (2 * sig ** 2))


def bimodal(x, std, mean, deltaF=0):
    """Forward bimodal distribution with DFT weighting."""
    shifted = np.asarray(x) - deltaF
    forward = gauss(shifted, std, mean)
    backward = np.exp(shifted) * gauss(-shifted, std, mean)
    result = np.where(shifted >= 0, forward, backward)
    if np.isscalar(x):
        return float(result)
    return result


def bimodal_rev(x, std, mean, deltaF=0):
    """Reverse bimodal distribution with DFT weighting."""
    shifted = np.asarray(x) + deltaF
    forward = gauss(shifted, std, mean)
    backward = np.exp(shifted) * gauss(-shifted, std, mean)
    result = np.where(shifted >= 0, forward, backward)
    if np.isscalar(x):
        return float(result)
    return result


def loggauss(x, sig, mu):
    """Gaussian evaluated in log space."""
    return -((x - mu) ** 2) / (2 * sig ** 2)


def log_bimodal(x, std, mean):
    """Log-space version of the bimodal distribution."""
    x_arr = np.asarray(x)
    positive = loggauss(x_arr, std, mean)
    negative = x_arr + loggauss(-x_arr, std, mean)
    out = np.where(x_arr >= 0, positive, negative)
    result = np.exp(out)
    if np.isscalar(x):
        return float(result)
    return result


def generate_dist(pdf_nonorm, args, kwargs):
    """Generate a ContinuousDist from an unnormalised pdf."""
    pdf = lambda x: pdf_nonorm(x, *args, **kwargs)
    dist = ContinuousDist()
    dist.set_pdf(pdf)
    return dist


def dist_from_piecewise_list(x_vals, pdf_vals):
    norm = np.trapz(pdf_vals, x_vals)
    dist = ContinuousDist()
    dist.norm = norm
    dist._pdf = lambda x: np.interp(x, x_vals, pdf_vals, left=0, right=0) / dist.norm
    pad = max((x_vals[-1] - x_vals[0]) * 0.05, 1e-3)
    dist.lims = [x_vals[0] - pad, x_vals[-1] + pad]
    return dist


def reverse_pdf(x, fpdf, deltaF):
    x_arr = np.asarray(x)
    pdf_vals = fpdf(-x_arr)
    safe_vals = np.where(pdf_vals == 0, 0.0, pdf_vals * np.exp(x_arr + deltaF))
    if np.isscalar(x):
        return float(safe_vals)
    return safe_vals


def rdist_from_piecesise_dist(dist):
    deltaF = -np.log(dist.negexp_avg())

    rdist = ContinuousDist()
    rdist.norm = dist.norm
    rdist._pdf = lambda x: reverse_pdf(x, dist.pdf, deltaF)
    rdist.lims = [-dist.lims[-1], -dist.lims[0]]
    return rdist


class ContinuousDist(sp.stats.rv_continuous):
    """Augmented scipy.stats continuous distribution."""

    def get_lims(self):
        return getattr(self, "lims", [-np.inf, np.inf])

    def set_pdf(self, pdf_func, lims=None):
        if lims is None:
            lims = self.get_lims()
        self.norm = sp.integrate.quad(pdf_func, *lims)[0]
        self.pdf_func = pdf_func
        self._pdf = lambda y: self.pdf_func(y) / self.norm

    def func_avg(self, kernel_lambda, lims=None):
        if lims is None:
            lims = self.get_lims()
        return sp.integrate.quad(kernel_lambda, *lims)[0]

    def get_min_eps(self, lims=None):
        kernel = lambda x: np.tanh(x / 2) * self.pdf(x)
        tanh_avg = self.func_avg(kernel, lims)
        return (1 / tanh_avg) - 1

    def mean(self, lims=None):
        kernel = lambda x: x * self.pdf(x)
        return self.func_avg(kernel, lims)

    def var(self, lims=None):
        mean = self.mean(lims)
        kernel = lambda x: (x - mean) ** 2 * self.pdf(x)
        return self.func_avg(kernel, lims)

    def moment(self, k, lims=None):
        mean = self.mean(lims)
        kernel = lambda x: (x - mean) ** k * self.pdf(x)
        return self.func_avg(kernel, lims)

    def negexp_avg(self, lims=None):
        kernel = lambda x: np.exp(-x) * self.pdf(x)
        return self.func_avg(kernel, lims)

    def negexp_avg_num(self, lims=None, resolution=10_000):
        def negexp(x):
            return np.exp(-x)

        return self.func_avg_num(negexp, lims, resolution=resolution)

    def func_avg_num(self, func, lims=None, resolution=10_000):
        if lims is None:
            lims = self.get_lims()
        x = np.linspace(*lims, resolution)
        pdf_x = [self.pdf(item) for item in x]
        return np.trapz(func(x) * pdf_x, dx=x[1] - x[0])

    def check_mean_num(self, lims, N=200_000):
        if lims is None:
            lims = self.get_lims()
        x = np.linspace(*lims, N)
        dx = x[1] - x[0]
        pdf_x = [self.pdf(item) for item in x]
        return dx * np.sum(x * pdf_x)

    def check_mean_sample(self, N):
        return np.mean(self.rvs(size=N))

    def analytic_trimodal_bound(E):
        return 2 * E * np.tanh(E / 2), (np.tanh(E) * np.tanh(E / 2)) ** -1 - 1

    def trimodal_eps(sigma_list, npoints=1000):
        ll = min(sigma_list)
        ul = max(sigma_list)
        if ll == ul:
            ul += 0.5
        avgs, bnd_list = ContinuousDist.analytic_trimodal_bound(np.linspace(0.1, ul, npoints))
        return UnivariateSpline(avgs, bnd_list, k=1, s=0)


class HistDist:
    def __init__(self, target_pdf, bins=None):
        self.target_pdf = target_pdf
        if bins is None:
            bins = np.linspace(-10, 10, 1_000)
        self.bins = bins
        self.counts, self.bins = self.get_histogram()
        self.histogram = self.counts, self.bins
        self.n_accepted, self.n_sampled = 0, 0
        self.verbose = False

    def target(self, x):
        try:
            return self.target_pdf(x)
        except (TypeError, ValueError):
            x_arr = np.asarray(x)
            return np.array([self.target_pdf(i) for i in x_arr])

    def get_histogram(self, density=False):
        counts = np.max((self.target(self.bins[:-1]), self.target(self.bins[1:])), axis=0)
        if density:
            counts = counts / np.sum(np.diff(self.bins) * counts)
        return counts, self.bins

    def normalize(self):
        self.counts = self.counts / np.sum(np.diff(self.bins) * self.counts)

    def pdf(self, x):
        def square_step(val, lims, height):
            return height * (np.heaviside(val - lims[0], 0) + np.heaviside(lims[1] - val, 1) - 1)

        return sum(square_step(x, self.bins[i:i + 2], self.counts[i]) for i in range(len(self.counts)))

    def naive_sample(self, N):
        probs = np.diff(self.bins) * self.counts
        indices = np.random.choice(range(len(self.counts)), size=N, p=probs / probs.sum())
        y = np.random.uniform(self.bins[indices], self.bins[indices + 1])
        prob_y = self.counts[indices]
        return y, prob_y

    def sample(self, N):
        counts, bins = self.histogram
        prob_bins = np.pad(np.cumsum(counts * np.diff(bins)), (1, 0))
        U = np.random.uniform(0, np.max(prob_bins), N)
        y = np.interp(U, prob_bins, bins)
        L, R = prob_bins[:-1], prob_bins[1:]
        prob_y = np.select([(U > l) & (U <= r) for l, r in zip(L, R)], counts)
        return y, prob_y

    def find_M(self, plot=False, max_iter=1_000):
        try:
            M = self.M
        except AttributeError:
            M = 1.0

        x = self.bins[1:-1]
        pdf_vals = self.pdf(x)
        target_vals = self.target(x)
        pdf_safe = np.where(pdf_vals > 0, pdf_vals, np.finfo(float).tiny)
        target_vals = np.nan_to_num(target_vals, nan=0.0, posinf=0.0, neginf=0.0)

        diff = M * pdf_vals - target_vals
        idx = int(np.argmin(diff))
        diff_min = diff[idx]

        if plot:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            c, b = self.histogram
            ax[0, 0].stairs(M * c, b, label='M$\cdot$proposal')
            ax[0, 0].plot(x, target_vals, label='target')
            ax[0, 0].set_title(f'prob density when M={M}')
            ax[0, 1].set_ylabel('M$\cdot$proposal-target')
            ax[0, 1].plot(x, diff, c='r')
            fig.legend()

        i = 0
        while (not np.isclose(diff_min, 0, atol=1e-6, rtol=1e-4) or diff_min < 0) and i <= max_iter:
            M_prev = M
            candidate = target_vals[idx] / pdf_safe[idx]
            if not np.isfinite(candidate) or candidate <= 0:
                candidate = M_prev * 1.05
            M = candidate
            if np.isclose(M, M_prev):
                M *= 1 + np.abs(np.random.normal(0, 1e-4))
            diff = M * pdf_vals - target_vals
            idx = int(np.argmin(diff))
            diff_min = diff[idx]
            print(f'M={M}, dff_min:{diff_min}', end='\r')
            i += 1

        if i > max_iter:
            print(f'failed to converge on M after {max_iter} tries')
        self.M = M

        if plot:
            ax[1, 0].stairs(M * self.histogram[0], self.histogram[1])
            ax[1, 0].set_title(f'prob density when M={M}')
            ax[1, 0].plot(x, target_vals)
            ax[1, 1].set_ylabel('M$\cdot$proposal-target')
            ax[1, 1].plot(x, diff)
            return (M, diff_min, fig)
        return (M, diff_min)

    def get_acceptance_ratio(self):
        try:
            return self.n_accepted / self.n_sampled
        except ZeroDivisionError:
            return None

    def rejection_sample(self, N):
        if self.get_acceptance_ratio() is None:
            _ = self.rejection_trial(500)
        ratio = self.get_acceptance_ratio()
        if ratio is None or ratio <= 0:
            ratio = 1.0
        num_samples = max(int(N / ratio), N)
        accepted_samples = self.rejection_trial(num_samples)
        if self.verbose and num_samples > 0:
            print(
                f'sample acceptance ratio:{len(accepted_samples) / max(num_samples, 1):.4f}, '
                f'expected:{ratio:.4f}'
            )
        return accepted_samples

    def rejection_trial(self, N):
        if not hasattr(self, 'M'):
            _ = self.find_M()
        y, prob_gy = self.sample(N)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = self.target(y) / (self.M * prob_gy)
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=1.0, neginf=0.0)
        ratio = np.clip(ratio, 0.0, 1.0)
        mask = np.random.uniform(0, 1, N) < ratio
        accepted_samples = y[mask]

        self.n_accepted += len(accepted_samples)
        self.n_sampled += N

        return accepted_samples


class RevHistDist(HistDist):
    def __init__(self, target_pdf, bins=None, deltaF=0):
        self.deltaF = deltaF
        self.target_pdf = lambda x: target_pdf(x - self.deltaF)
        super().__init__(self.target_pdf, bins)
