"""
fmcw_simulator.py
─────────────────────────────────────────────────────────────────────────────
FMCW Automotive Radar — Range-Doppler Map Simulator
77 GHz, configurable targets, heterogeneous noise backgrounds.

Output: complex RD map + magnitude RD map + target mask
─────────────────────────────────────────────────────────────────────────────
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Radar Parameters ──────────────────────────────────────────────────────────
@dataclass
class RadarParams:
    fc:         float = 77e9        # Carrier frequency [Hz]
    B:          float = 200e6       # Chirp bandwidth [Hz]
    T_chirp:    float = 40e-6       # Chirp duration [s]
    N_fast:     int   = 256         # Samples per chirp (range FFT size)
    N_slow:     int   = 128         # Chirps per frame (Doppler FFT size)
    fs:         float = 10e6        # IF sampling rate [Hz]

    @property
    def c(self):          return 3e8
    @property
    def lambda_(self):    return self.c / self.fc
    @property
    def range_res(self):  return self.c / (2 * self.B)
    @property
    def range_max(self):  return self.c * self.N_fast / (4 * self.B)
    @property
    def vel_res(self):    return self.lambda_ / (2 * self.N_slow * self.T_chirp)
    @property
    def vel_max(self):    return self.lambda_ / (4 * self.T_chirp)


@dataclass
class Target:
    range_m:    float          # Target range [m]
    velocity_ms:float          # Radial velocity [m/s]
    rcs_db:     float = 0.0    # Relative RCS [dB above reference]


@dataclass
class SceneConfig:
    targets:    List[Target]   = field(default_factory=list)
    snr_db:     float          = 15.0     # Mean target SNR [dB]
    noise_type: str            = 'rayleigh'   # 'rayleigh' | 'weibull' | 'lognormal'
    clutter_edges: bool        = False    # Add clutter edge in range
    multi_target_density: str  = 'low'   # 'low' | 'medium' | 'high'
    seed:       Optional[int]  = 42


class FMCWSimulator:
    """
    Generates a 2D Range-Doppler magnitude map for a given scene configuration.
    Target signal is injected as a complex sinusoid at the correct (range, Doppler) bin.
    Noise is drawn from the specified distribution.
    """
    def __init__(self, params: RadarParams = None):
        self.p = params or RadarParams()

    # ── Core RD-map generator ─────────────────────────────────────────────────
    def generate(self, scene: SceneConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        rd_power   : (N_range, N_doppler) float array — power (linear scale)
        target_mask: (N_range, N_doppler) bool array  — True at target cells
        """
        rng  = np.random.default_rng(scene.seed)
        Nr   = self.p.N_fast // 2   # One-sided range FFT
        Nd   = self.p.N_slow
        snr  = 10 ** (scene.snr_db / 10)

        # ── Noise floor ───────────────────────────────────────────────────────
        noise_power = self._gen_noise(rng, Nr, Nd, scene)

        # ── Target power ──────────────────────────────────────────────────────
        signal_power = np.zeros((Nr, Nd), dtype=float)
        target_mask  = np.zeros((Nr, Nd), dtype=bool)

        for t in scene.targets:
            ri = self._range_bin(t.range_m)
            di = self._doppler_bin(t.velocity_ms)
            if 0 <= ri < Nr and 0 <= di < Nd:
                # Local noise power at that cell
                cell_noise = noise_power[ri, di]
                amp = np.sqrt(snr * cell_noise * (10 ** (t.rcs_db / 10)))
                # Spread over main lobe ±1 bin with Hann weighting
                for dr in range(-1, 2):
                    for dd in range(-1, 2):
                        r2, d2 = ri + dr, (di + dd) % Nd
                        if 0 <= r2 < Nr:
                            w = (0.5 + 0.5 * np.cos(dr * np.pi / 2)) * \
                                (0.5 + 0.5 * np.cos(dd * np.pi / 2))
                            signal_power[r2, d2] += (amp * w) ** 2
                target_mask[ri, di] = True

        rd_power = noise_power + signal_power
        return rd_power.astype(np.float32), target_mask

    # ── Noise generation ──────────────────────────────────────────────────────
    def _gen_noise(self, rng, Nr, Nd, scene):
        base = np.ones((Nr, Nd), dtype=float)

        if scene.noise_type == 'rayleigh':
            # Exponential power (Rayleigh amplitude)
            noise = rng.exponential(scale=1.0, size=(Nr, Nd))
        elif scene.noise_type == 'weibull':
            k, lam = 0.8, 1.2
            noise = (rng.weibull(k, size=(Nr, Nd)) * lam) ** 2
        elif scene.noise_type == 'lognormal':
            noise = rng.lognormal(mean=0.0, sigma=0.6, size=(Nr, Nd))
        else:
            noise = rng.exponential(1.0, size=(Nr, Nd))

        # Clutter edge: range bins > Nr//2 have 10 dB higher noise floor
        if scene.clutter_edges:
            edge = Nr // 2
            noise[edge:, :] *= 10.0

        # Multi-target background clutter patches
        if scene.multi_target_density in ('medium', 'high'):
            n_clutter = 8 if scene.multi_target_density == 'medium' else 20
            for _ in range(n_clutter):
                ri = rng.integers(0, Nr)
                di = rng.integers(0, Nd)
                noise[max(0,ri-2):ri+3, max(0,di-2):di+3] *= rng.uniform(3, 8)

        return base * noise

    # ── Bin conversion ────────────────────────────────────────────────────────
    def _range_bin(self, r_m):
        return int(r_m / self.p.range_res)

    def _doppler_bin(self, v_ms):
        fd = 2 * v_ms / self.p.lambda_
        bin_idx = int(fd / (1 / (self.p.N_slow * self.p.T_chirp)))
        return int(bin_idx % self.p.N_slow)


# ── Convenience: generate standard evaluation scenes ─────────────────────────
def make_scene_suite(seed_offset=0):
    """Returns a list of (scene_name, SceneConfig) for benchmarking."""
    p = RadarParams()
    scenes = []

    for snr in [5, 10, 15, 20, 25]:
        targets = [
            Target(range_m=30, velocity_ms=15),
            Target(range_m=60, velocity_ms=-10),
            Target(range_m=90, velocity_ms=5),
        ]
        scenes.append((f"homogeneous_SNR{snr}dB", SceneConfig(
            targets=targets, snr_db=snr, noise_type='rayleigh',
            clutter_edges=False, seed=100 + snr + seed_offset)))

    for nt in ['rayleigh', 'weibull', 'lognormal']:
        targets = [Target(range_m=50, velocity_ms=20)]
        scenes.append((f"noise_{nt}_SNR15dB", SceneConfig(
            targets=targets, snr_db=15, noise_type=nt,
            clutter_edges=False, seed=200 + seed_offset)))

    # Heterogeneous scenes
    scenes.append(("clutter_edge_SNR15dB", SceneConfig(
        targets=[Target(range_m=50, velocity_ms=20), Target(range_m=80, velocity_ms=-5)],
        snr_db=15, noise_type='rayleigh', clutter_edges=True, seed=300 + seed_offset)))

    scenes.append(("multi_target_dense_SNR20dB", SceneConfig(
        targets=[Target(range_m=r, velocity_ms=v)
                 for r, v in [(20,10),(35,-5),(50,20),(65,0),(80,-15)]],
        snr_db=20, noise_type='rayleigh',
        multi_target_density='high', seed=400 + seed_offset)))

    return scenes


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim = FMCWSimulator()
    scene = SceneConfig(
        targets=[Target(30, 15), Target(60, -10), Target(90, 5)],
        snr_db=15,
        noise_type='rayleigh',
        clutter_edges=True,
        seed=42
    )

    rd, mask = sim.generate(scene)

    print(f"RD map shape: {rd.shape}, power range: [{rd.min():.3f}, {rd.max():.3f}]")
    print(f"Target cells: {mask.sum()}")

    # Display the simulated Range-Doppler power map
    range_axis = np.arange(rd.shape[0]) * sim.p.range_res
    velocity_axis = (np.arange(rd.shape[1]) - rd.shape[1] // 2) * sim.p.vel_res

    rd_shifted = np.fft.fftshift(rd, axes=1)
    mask_shifted = np.fft.fftshift(mask, axes=1)

    plt.figure(figsize=(9, 5))
    plt.imshow(
        10 * np.log10(rd_shifted + 1e-12),
        extent=[velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]],
        aspect='auto',
        origin='lower'
    )
    target_r, target_d = np.where(mask_shifted)
    plt.scatter(
        velocity_axis[target_d],
        range_axis[target_r],
        marker='x',
        s=80,
        linewidths=2,
        label='True target'
    )
    plt.xlabel('Radial Velocity (m/s)')
    plt.ylabel('Range (m)')
    plt.title('77 GHz FMCW Radar – Simulated Range-Doppler Map')
    plt.colorbar(label='Power (dB)')
    plt.legend()
    plt.tight_layout()
    plt.show()

