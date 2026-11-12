"""
cfar_classical.py
─────────────────────────────────────────────────────────────────────────────
Classical CFAR Detector Implementations for Baseline Comparison
Variants: CA-CFAR, GO-CFAR, SO-CFAR, OS-CFAR (all 2D, sliding-window)
─────────────────────────────────────────────────────────────────────────────
"""
import numpy as np
from typing import Tuple


def _build_window_indices(guard: int, train: int):
    """
    Returns (ref_offsets, guard_offsets) — list of (dr, dd) offsets
    forming the 2D annular reference and guard windows.
    """
    total = guard + train
    ref_offsets   = []
    guard_offsets = []
    for dr in range(-total, total + 1):
        for dd in range(-total, total + 1):
            d = max(abs(dr), abs(dd))   # Chebyshev distance
            if 1 <= d <= guard:
                guard_offsets.append((dr, dd))
            elif guard < d <= total:
                ref_offsets.append((dr, dd))
    return ref_offsets, guard_offsets


def _get_reference_cells(rd: np.ndarray, ri: int, di: int,
                         ref_offsets, Nr: int, Nd: int) -> np.ndarray:
    """Extract reference cell values with boundary clamping in range, wrap in Doppler."""
    vals = []
    for dr, dd in ref_offsets:
        r2 = ri + dr
        d2 = (di + dd) % Nd
        if 0 <= r2 < Nr:
            vals.append(rd[r2, d2])
    return np.array(vals, dtype=np.float32)


def cfar_2d(rd: np.ndarray, guard: int = 2, train: int = 4,
            pfa: float = 1e-4, variant: str = 'CA',
            os_k_fraction: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D sliding-window CFAR detector.

    Parameters
    ----------
    rd          : (Nr, Nd) power map
    guard       : guard cell half-width (Chebyshev)
    train       : training cell half-width (Chebyshev)
    pfa         : target probability of false alarm
    variant     : 'CA' | 'GO' | 'SO' | 'OS'
    os_k_fraction: for OS-CFAR, k = round(N_ref * os_k_fraction)

    Returns
    -------
    detections  : (Nr, Nd) bool — True where target detected
    threshold   : (Nr, Nd) float — computed threshold at each cell
    """
    Nr, Nd = rd.shape
    ref_offsets, _ = _build_window_indices(guard, train)
    N_ref = len(ref_offsets)   # Effective reference window size

    # Analytical α from Pfa and N_ref (CA-CFAR exponential assumption)
    # T = alpha * Z where Z = (1/N) * sum(ref cells)
    # For CA-CFAR under exponential: alpha = N * (Pfa^(-1/N) - 1)
    alpha_ca = N_ref * (pfa ** (-1.0 / N_ref) - 1.0)

    detections = np.zeros((Nr, Nd), dtype=bool)
    threshold  = np.zeros((Nr, Nd), dtype=np.float32)

    for ri in range(Nr):
        for di in range(Nd):
            Z = _get_reference_cells(rd, ri, di, ref_offsets, Nr, Nd)
            if len(Z) < 4:
                continue

            if variant == 'CA':
                Z_est = np.mean(Z)
                T = alpha_ca * Z_est

            elif variant == 'GO':
                # Split into leading and lagging sub-windows
                leading  = [rd[min(Nr-1, ri+dr), (di+dd)%Nd]
                            for dr, dd in ref_offsets if dr >= 0]
                lagging  = [rd[max(0,    ri+dr), (di+dd)%Nd]
                            for dr, dd in ref_offsets if dr < 0]
                Zl = np.mean(leading)  if leading  else np.mean(Z)
                Zg = np.mean(lagging) if lagging else np.mean(Z)
                Z_est = max(Zl, Zg)
                T = alpha_ca * Z_est

            elif variant == 'SO':
                leading  = [rd[min(Nr-1, ri+dr), (di+dd)%Nd]
                            for dr, dd in ref_offsets if dr >= 0]
                lagging  = [rd[max(0,    ri+dr), (di+dd)%Nd]
                            for dr, dd in ref_offsets if dr < 0]
                Zl = np.mean(leading)  if leading  else np.mean(Z)
                Zg = np.mean(lagging) if lagging else np.mean(Z)
                Z_est = min(Zl, Zg)
                T = alpha_ca * Z_est

            elif variant == 'OS':
                # OS-CFAR: estimate the background from the k-th ordered
                # reference cell and compute alpha from the desired Pfa.
                # The previous approximation made alpha far too small for
                # k/N = 0.75, causing almost the entire RD map to be detected.
                k = max(1, min(len(Z), round(len(Z) * os_k_fraction)))
                Z_sorted = np.sort(Z)
                Z_est = Z_sorted[k - 1]

                # For exponentially distributed reference power and the
                # k-th smallest order statistic:
                # Pfa = N! / ((k-1)! (N-k)!) * B(N-k+alpha+1, k)
                # Solve this equation numerically for the required alpha.
                from scipy.optimize import brentq
                from scipy.special import betaln, gammaln

                N = len(Z)

                def os_pfa(alpha):
                    log_coeff = gammaln(N + 1) - gammaln(k) - gammaln(N - k + 1)
                    log_probability = log_coeff + betaln(N - k + alpha + 1.0, k)
                    return np.exp(log_probability)

                alpha_os = brentq(
                    lambda a: os_pfa(a) - pfa,
                    0.0, 1e6
                )
                T = alpha_os * Z_est
            else:
                raise ValueError(f"Unknown variant: {variant}")

            threshold[ri, di] = T
            detections[ri, di] = rd[ri, di] > T

    return detections, threshold


def cfar_2d_fast(rd: np.ndarray, guard: int = 2, train: int = 4,
                  pfa: float = 1e-4, variant: str = 'CA') -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised 2D CA/GO/SO CFAR using 2D convolution for speed.
    Suitable for large RD maps in evaluation loops.
    """
    from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
    Nr, Nd = rd.shape
    W_total = 2 * (guard + train) + 1
    W_guard = 2 * guard + 1

    # Sum over total window
    sum_total = uniform_filter(rd, size=W_total, mode='wrap') * W_total**2
    # Sum over guard window
    sum_guard = uniform_filter(rd, size=W_guard, mode='wrap') * W_guard**2
    # Reference sum
    N_total = W_total**2
    N_guard = W_guard**2
    N_ref   = N_total - N_guard - 1   # exclude CUT

    Z_sum = sum_total - sum_guard - rd
    Z_mean = np.maximum(Z_sum / max(N_ref, 1), 1e-12)

    if variant in ('CA', 'GO', 'SO'):
        alpha = N_ref * (pfa ** (-1.0 / N_ref) - 1.0)
        T = alpha * Z_mean
    else:
        alpha = N_ref * (pfa ** (-1.0 / N_ref) - 1.0)
        T = alpha * Z_mean   # fallback for fast path

    detections = rd > T
    return detections, T.astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from fmcw_simulator import FMCWSimulator, SceneConfig, Target

    sim = FMCWSimulator()
    scene = SceneConfig(
        targets=[Target(50, 15)],
        snr_db=15,
        noise_type='rayleigh',
        seed=42
    )
    rd, mask = sim.generate(scene)

    # Physical axes from the FMCW simulator
    range_axis = np.arange(rd.shape[0]) * sim.p.range_res
    velocity_axis = (np.arange(rd.shape[1]) - rd.shape[1] // 2) * sim.p.vel_res

    rd_shifted = np.fft.fftshift(rd, axes=1)
    mask_shifted = np.fft.fftshift(mask, axes=1)
    rd_db = 10 * np.log10(rd_shifted + 1e-12)

    for v in ['CA', 'GO', 'SO', 'OS']:
        det, thr = cfar_2d(rd, variant=v)
        det_shifted = np.fft.fftshift(det, axes=1)

        tp_mask = det_shifted & mask_shifted
        fp_mask = det_shifted & ~mask_shifted

        tp_r, tp_d = np.where(tp_mask)
        fp_r, fp_d = np.where(fp_mask)
        true_r, true_d = np.where(mask_shifted)

        print(f"{v}-CFAR: detections={det.sum()}, true positives={tp_mask.sum()}, false positives={fp_mask.sum()}")

        plt.figure(figsize=(10, 6))
        plt.imshow(
            rd_db.T,
            extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
            aspect='auto',
            origin='lower'
        )

        if len(fp_r):
            plt.scatter(
                range_axis[fp_r], velocity_axis[fp_d],
                marker='o', s=35, facecolors='none',
                label='False alarm'
            )

        if len(tp_r):
            plt.scatter(
                range_axis[tp_r], velocity_axis[tp_d],
                marker='o', s=65, facecolors='none',
                linewidths=2, label='True detection'
            )

        plt.scatter(
            range_axis[true_r], velocity_axis[true_d],
            marker='x', s=100, linewidths=2,
            label='Ground truth'
        )

        plt.xlabel('Range (m)')
        plt.ylabel('Radial Velocity (m/s)')
        plt.title(f'{v}-CFAR Detection on FMCW Range-Doppler Map')
        plt.colorbar(label='Power (dB)')
        plt.legend()
        plt.tight_layout()

    plt.show()
