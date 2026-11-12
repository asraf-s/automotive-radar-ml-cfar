"""
alpha_labeler_v2.py
─────────────────────────────────────────────────────────────────────────────
Corrected ground-truth α labelling via non-parametric bootstrap P_FA
calibration.

Key fix vs v1: CUT samples are bootstrapped from the ACTUAL reference window
distribution rather than an exponential fit, so α* genuinely varies with the
local noise shape (kurtosis, skewness, heavy tails) and the regression task
is meaningful.
─────────────────────────────────────────────────────────────────────────────
"""
import numpy as np
from scipy.optimize import brentq
from typing import Tuple


FEATURE_NAMES = [
    "mean_power",      # μ
    "variance",        # σ²
    "skewness",        # γ₁
    "kurtosis",        # γ₂ (excess)
    "IQR",             # Q75−Q25
    "max_mean_ratio",  # max/μ
    "lead_lag_ratio",  # μ_lead / μ_lag
    "coeff_variation", # σ/μ
]


def extract_features(Z: np.ndarray) -> np.ndarray:
    Z = Z[Z > 0]
    if len(Z) < 4:
        return np.zeros(8, dtype=np.float32)
    mu    = float(np.mean(Z))
    sig   = float(np.std(Z)) + 1e-12
    n     = len(Z)
    skew  = float(np.mean(((Z - mu)/sig)**3))
    kurt  = float(np.mean(((Z - mu)/sig)**4)) - 3.0
    q75,q25 = np.percentile(Z,[75,25])
    iqr   = float(q75 - q25)
    mm    = float(np.max(Z)) / (mu + 1e-12)
    half  = n // 2
    ll    = float(np.mean(Z[:half])) / (float(np.mean(Z[half:])) + 1e-12) if half>0 else 1.0
    cv    = sig / (mu + 1e-12)
    return np.array([mu, sig**2, skew, kurt, iqr, mm, ll, cv], dtype=np.float32)


def bootstrap_pfa(alpha: float, Z: np.ndarray, n_mc: int, rng) -> float:
    """
    Non-parametric bootstrap estimate of P_FA.
    Under H₀ (noise only), the CUT is drawn from the same distribution as Z.
    Reference windows are also bootstrapped from Z.
    """
    N = len(Z)
    if N < 4:
        return float(alpha > N * (1e-4 ** (-1.0/max(N,1)) - 1.0))
    # Bootstrap CUT samples from Z
    cut_samples = rng.choice(Z, size=n_mc, replace=True)
    # Bootstrap reference windows
    ref_samples = rng.choice(Z, size=(n_mc, N), replace=True)
    Z_mean      = ref_samples.mean(axis=1)
    thresholds  = alpha * Z_mean
    return float(np.mean(cut_samples > thresholds))


def find_alpha_star(Z: np.ndarray, target_pfa: float = 1e-4,
                    n_mc: int = 4000, seed: int = 0) -> float:
    """Find α* via Brent's method on the bootstrap P_FA function."""
    N   = len(Z)
    rng = np.random.default_rng(seed)
    # Analytical CA-CFAR reference point
    alpha_ca = N * (target_pfa ** (-1.0/max(N,1)) - 1.0)
    if N < 4:
        return alpha_ca
    try:
        f  = lambda a: bootstrap_pfa(a, Z, n_mc, rng) - target_pfa
        lo = max(0.5, alpha_ca * 0.15)
        hi = alpha_ca * 8.0
        flo, fhi = f(lo), f(hi)
        # If same sign, use fallback
        if flo * fhi > 0:
            return alpha_ca
        return float(brentq(f, lo, hi, xtol=1e-2, maxiter=25))
    except Exception:
        return alpha_ca


def label_rd_map(rd: np.ndarray, guard: int = 2, train: int = 4,
                 target_pfa: float = 1e-4, n_mc: int = 3000,
                 subsample: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Label all valid CUTs in the RD map.
    Returns (X, y): feature matrix and α* labels.
    """
    Nr, Nd    = rd.shape
    W_total   = guard + train + 1
    W_guard   = guard + 1
    feats, labels = [], []

    for ri in range(W_total, Nr - W_total, subsample):
        for di in range(0, Nd, subsample):
            Z = []
            for dr in range(-W_total, W_total+1):
                for dd in range(-W_total, W_total+1):
                    d = max(abs(dr), abs(dd))
                    if W_guard <= d <= W_total:
                        r2 = ri + dr
                        d2 = (di + dd) % Nd
                        if 0 <= r2 < Nr:
                            Z.append(rd[r2, d2])
            Z_arr = np.array(Z, dtype=np.float32)
            feats.append(extract_features(Z_arr))
            labels.append(find_alpha_star(Z_arr, target_pfa=target_pfa,
                                           n_mc=n_mc, seed=ri*Nd+di))
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.float32)
