"""
alpha_labeler.py + feature_extractor.py
─────────────────────────────────────────────────────────────────────────────
Ground-truth α Generation via Monte Carlo P_FA Calibration
and 8-Dimensional Local Noise Statistics Feature Extraction
─────────────────────────────────────────────────────────────────────────────
"""
import numpy as np
from scipy.optimize import brentq
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_cell(Z: np.ndarray) -> np.ndarray:
    """
    Extract 8-dimensional local noise statistics from a reference window Z.

    Features
    --------
    f0 : mean power (μ)
    f1 : variance (σ²)
    f2 : skewness (γ₁)
    f3 : excess kurtosis (γ₂)
    f4 : inter-quartile range (IQR = Q75 - Q25)
    f5 : max-to-mean ratio (heterogeneity index)
    f6 : leading/lagging mean ratio (clutter-edge indicator)
    f7 : coefficient of variation (σ/μ)
    """
    Z = Z[Z > 0]   # guard against zeros
    if len(Z) < 4:
        return np.zeros(8, dtype=np.float32)

    mu      = float(np.mean(Z))
    sigma2  = float(np.var(Z))
    sigma   = np.sqrt(sigma2 + 1e-12)
    n       = len(Z)

    # Skewness
    skew = float(np.mean(((Z - mu) / sigma) ** 3)) if sigma > 0 else 0.0
    # Excess kurtosis
    kurt = float(np.mean(((Z - mu) / sigma) ** 4)) - 3.0 if sigma > 0 else 0.0
    # IQR
    q75, q25 = np.percentile(Z, [75, 25])
    iqr = float(q75 - q25)
    # Max-to-mean ratio
    max_mean = float(np.max(Z)) / (mu + 1e-12)
    # Leading/lagging split (first half vs second half — proxy for clutter edge)
    half = n // 2
    lead_mean = float(np.mean(Z[:half])) if half > 0 else mu
    lag_mean  = float(np.mean(Z[half:])) if half > 0 else mu
    ll_ratio  = lead_mean / (lag_mean + 1e-12)
    # CV
    cv = sigma / (mu + 1e-12)

    return np.array([mu, sigma2, skew, kurt, iqr, max_mean, ll_ratio, cv],
                    dtype=np.float32)


def extract_all_features(rd: np.ndarray, guard: int = 2, train: int = 4,
                          subsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for every (ri, di) cell in the RD map.

    Parameters
    ----------
    rd         : (Nr, Nd) power map
    guard      : guard half-width (Chebyshev)
    train      : training half-width (Chebyshev)
    subsample  : process every `subsample`-th cell (for speed in training)

    Returns
    -------
    features   : (M, 8) float32
    cell_indices: (M, 2) int — (ri, di) for each row
    """
    Nr, Nd     = rd.shape
    W_guard    = guard + 1
    W_total    = guard + train + 1
    feats, idxs = [], []

    ri_range = range(W_total, Nr - W_total, subsample)
    di_range = range(0, Nd, subsample)

    for ri in ri_range:
        for di in di_range:
            # Annular reference window
            Z = []
            for dr in range(-W_total, W_total + 1):
                for dd in range(-W_total, W_total + 1):
                    d = max(abs(dr), abs(dd))
                    if W_guard <= d <= W_total:
                        r2 = ri + dr
                        d2 = (di + dd) % Nd
                        if 0 <= r2 < Nr:
                            Z.append(rd[r2, d2])
            Z = np.array(Z, dtype=np.float32)
            feats.append(extract_features_cell(Z))
            idxs.append((ri, di))

    return np.array(feats, dtype=np.float32), np.array(idxs, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — MONTE CARLO α LABELLING
# ═══════════════════════════════════════════════════════════════════════════════

def empirical_pfa(alpha: float, Z: np.ndarray, n_mc: int = 5000,
                  seed: int = 0) -> float:
    """
    Estimate P_FA for a given α by simulating noise-only hypotheses.
    The noise distribution is fitted empirically to Z via moment matching
    (exponential scale = mean(Z) for Rayleigh/exponential-power assumption).
    """
    rng      = np.random.default_rng(seed)
    mu_z     = float(np.mean(Z)) if len(Z) > 0 else 1.0
    # Draw noise-only CUT samples from same exponential distribution
    cut_noise = rng.exponential(scale=mu_z, size=n_mc)
    # For each MC trial, draw a fresh reference window and compute threshold
    N_ref     = len(Z)
    ref_noise = rng.exponential(scale=mu_z, size=(n_mc, N_ref))
    Z_mean    = np.mean(ref_noise, axis=1)
    T         = alpha * Z_mean
    pfa_hat   = float(np.mean(cut_noise > T))
    return pfa_hat


def find_optimal_alpha(Z: np.ndarray, target_pfa: float = 1e-4,
                       n_mc: int = 3000, seed: int = 0) -> float:
    """
    Binary search for α* such that empirical_pfa(α*, Z) ≈ target_pfa.
    Falls back to the analytical CA-CFAR formula if search fails.
    """
    N_ref = len(Z)
    if N_ref < 4:
        return N_ref * (target_pfa ** (-1.0 / max(N_ref, 1)) - 1.0)

    # Analytical initial guess
    alpha_analytical = N_ref * (target_pfa ** (-1.0 / N_ref) - 1.0)

    try:
        f = lambda a: empirical_pfa(a, Z, n_mc=n_mc, seed=seed) - target_pfa
        # Search in [0.1 * analytical, 10 * analytical]
        lo, hi = alpha_analytical * 0.1, alpha_analytical * 10.0
        # Check signs
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            return alpha_analytical
        alpha_opt = brentq(f, lo, hi, xtol=1e-3, maxiter=30)
        return float(alpha_opt)
    except Exception:
        return alpha_analytical


def label_alpha_map(rd: np.ndarray, guard: int = 2, train: int = 4,
                    target_pfa: float = 1e-4, n_mc: int = 2000,
                    subsample: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute α* for every valid cell in the RD map.

    Returns
    -------
    alpha_map  : (Nr, Nd) float32 — α* at each cell (0 = not computed)
    features   : (M, 8) float32
    labels     : (M,)   float32 — α* labels
    """
    Nr, Nd       = rd.shape
    alpha_map    = np.zeros((Nr, Nd), dtype=np.float32)
    W_guard      = guard + 1
    W_total      = guard + train + 1
    feats, labels = [], []

    ri_range = range(W_total, Nr - W_total, subsample)
    di_range = range(0, Nd, subsample)

    for ri in ri_range:
        for di in di_range:
            Z = []
            for dr in range(-W_total, W_total + 1):
                for dd in range(-W_total, W_total + 1):
                    d = max(abs(dr), abs(dd))
                    if W_guard <= d <= W_total:
                        r2 = ri + dr
                        d2 = (di + dd) % Nd
                        if 0 <= r2 < Nr:
                            Z.append(rd[r2, d2])
            Z = np.array(Z, dtype=np.float32)
            feat = extract_features_cell(Z)
            alpha_opt = find_optimal_alpha(Z, target_pfa=target_pfa,
                                           n_mc=n_mc, seed=ri * Nd + di)
            alpha_map[ri, di] = alpha_opt
            feats.append(feat)
            labels.append(alpha_opt)

    return alpha_map, np.array(feats, dtype=np.float32), np.array(labels, dtype=np.float32)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
    from fmcw_simulator import FMCWSimulator, SceneConfig, Target

    sim = FMCWSimulator()
    scene = SceneConfig(targets=[Target(50, 15)], snr_db=15,
                        noise_type='rayleigh', clutter_edges=True, seed=42)
    rd, mask = sim.generate(scene)
    print(f"RD map: {rd.shape}")

    feats, idxs = extract_all_features(rd, subsample=4)
    print(f"Features: {feats.shape}, sample:\n{feats[0]}")

    alpha_map, F, L = label_alpha_map(rd, subsample=8, n_mc=500)
    print(f"Labels: {L.shape}, mean α={L.mean():.3f}, std={L.std():.3f}")
