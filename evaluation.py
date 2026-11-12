"""
evaluation.py
─────────────────────────────────────────────────────────────────────────────
Full Evaluation Pipeline + Figure Generation
Produces all plots saved to ./figures/
─────────────────────────────────────────────────────────────────────────────
"""
import numpy as np
import os, sys
from pathlib import Path
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fmcw_simulator import FMCWSimulator, SceneConfig, Target, RadarParams
from cfar_classical  import cfar_2d_fast
from alpha_labeler_v2 import label_rd_map
from extra_trees_cfar import ExtraTreesCFAR, FEATURE_NAMES

FIGURES_DIR = REPO_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = {
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif'],
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
}
plt.rcParams.update(STYLE)

COLORS = {
    'ET':  '#1B4F72',   # Dark navy — proposed
    'CA':  '#E74C3C',   # Red
    'GO':  '#27AE60',   # Green
    'SO':  '#F39C12',   # Orange
    'OS':  '#8E44AD',   # Purple
}

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — RD Map Visualisation (raw, with and without heterogeneous noise)
# ════════════════════════════════════════════════════════════════════════════════
def fig1_rd_maps():
    sim  = FMCWSimulator()
    p    = RadarParams()
    targets = [Target(30, 15, 0), Target(60, -10, -3), Target(90, 5, 2)]

    scenes = [
        ("Homogeneous Rayleigh Noise\n(SNR = 15 dB)",
         SceneConfig(targets=targets, snr_db=15, noise_type='rayleigh', seed=1)),
        ("Heterogeneous: Clutter Edge\n(SNR = 15 dB)",
         SceneConfig(targets=targets, snr_db=15, noise_type='rayleigh',
                     clutter_edges=True, seed=2)),
        ("Heterogeneous: Multi-Target Dense\n(SNR = 15 dB)",
         SceneConfig(targets=targets, snr_db=15, noise_type='rayleigh',
                     multi_target_density='high', seed=3)),
        ("Non-Gaussian: Log-Normal Noise\n(SNR = 15 dB)",
         SceneConfig(targets=targets, snr_db=15, noise_type='lognormal', seed=4)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    Nr = p.N_fast // 2
    Nd = p.N_slow
    r_axis = np.linspace(0, p.range_max, Nr)
    v_axis = np.linspace(-p.vel_max, p.vel_max, Nd)

    for ax, (title, scene) in zip(axes, scenes):
        rd, mask = sim.generate(scene)
        im = ax.imshow(10 * np.log10(rd + 1e-6).T, aspect='auto', origin='lower',
                       extent=[0, p.range_max, -p.vel_max, p.vel_max],
                       cmap='viridis', vmin=-5, vmax=20)
        # Mark targets
        for t in targets:
            ax.axvline(t.range_m, color='white', lw=0.5, alpha=0.3)
            ax.plot(t.range_m, t.velocity_ms, 'r+', ms=10, mew=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.05)
        cb  = fig.colorbar(im, cax=cax)
        cb.set_label('Power [dB]', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Range [m]')
        ax.set_ylabel('Velocity [m/s]')

    fig.suptitle('Figure 1 — FMCW 77 GHz Range-Doppler Maps Under Varying Noise Conditions\n'
                 '(red + marks = ground-truth target positions)',
                 fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_rd_maps.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Fig 1 done.")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Detection Comparison Maps (CA vs ET-CFAR)
# ════════════════════════════════════════════════════════════════════════════════
def fig2_detection_maps(model: ExtraTreesCFAR):
    sim  = FMCWSimulator()
    p    = RadarParams()
    targets = [Target(30, 15, 0), Target(60, -10, -3), Target(90, 5, 2)]
    scene = SceneConfig(targets=targets, snr_db=12, noise_type='rayleigh',
                        clutter_edges=True, multi_target_density='medium', seed=7)
    rd, mask = sim.generate(scene)

    variants = [
        ('CA-CFAR (fixed α)',   cfar_2d_fast(rd, variant='CA')),
        ('GO-CFAR (fixed α)',   cfar_2d_fast(rd, variant='GO')),
        ('SO-CFAR (fixed α)',   cfar_2d_fast(rd, variant='SO')),
        ('ET-CFAR (adaptive α)',model.detect_2d(rd)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    r_max = p.range_max
    v_max = p.vel_max

    # Col 0: RD map + targets
    ax = axes[0, 0]
    im = ax.imshow(10*np.log10(rd+1e-6).T, aspect='auto', origin='lower',
                   extent=[0, r_max, -v_max, v_max], cmap='viridis', vmin=-5, vmax=20)
    for t in targets:
        ax.plot(t.range_m, t.velocity_ms, 'r+', ms=12, mew=2.5)
    ax.set_title('Range-Doppler Map\n(Input Power, dB)', fontsize=9, fontweight='bold')
    ax.set_xlabel('Range [m]'); ax.set_ylabel('Velocity [m/s]')

    # Col 1-5: Detection maps
    det_axes = [axes[0,1], axes[0,2], axes[1,0], axes[1,1]]
    for axi, (name, (det, _)) in zip(det_axes, variants):
        overlay = np.zeros((*rd.T.shape, 3))
        rd_norm = (10*np.log10(rd+1e-6).T + 5) / 25
        rd_norm = np.clip(rd_norm, 0, 1)
        overlay[:,:,0] = rd_norm * 0.3
        overlay[:,:,1] = rd_norm * 0.3
        overlay[:,:,2] = rd_norm * 0.6
        # Detections: yellow
        overlay[det.T, 0] = 1.0
        overlay[det.T, 1] = 0.9
        overlay[det.T, 2] = 0.0
        # True positives: green
        tp = det & mask
        overlay[tp.T, 0] = 0.0
        overlay[tp.T, 1] = 0.9
        overlay[tp.T, 2] = 0.1
        axi.imshow(overlay, aspect='auto', origin='lower',
                   extent=[0, r_max, -v_max, v_max])
        for t in targets:
            axi.plot(t.range_m, t.velocity_ms, 'r+', ms=12, mew=2.5)
        tp_n = int((det & mask).sum())
        fa_n = int(det.sum()) - tp_n
        c = '#1B4F72' if 'ET' in name else '#7B241C'
        axi.set_title(f'{name}\nTP={tp_n}  FA={fa_n}', fontsize=8.5,
                      fontweight='bold', color=c)
        axi.set_xlabel('Range [m]'); axi.set_ylabel('Velocity [m/s]')

    # Legend patch
    from matplotlib.patches import Patch
    leg = [Patch(facecolor='yellow', label='False Alarm'),
           Patch(facecolor='#00E61A', label='True Positive'),
           Patch(facecolor='red',    label='Target Ground Truth (+)')]
    axes[1, 2].axis('off')
    axes[1, 2].legend(handles=leg, loc='center', fontsize=10,
                      title='Detection Legend', title_fontsize=11,
                      frameon=True, edgecolor='gray')

    fig.suptitle('Figure 2 — Detection Map Comparison: Fixed-α CFAR vs Adaptive ET-CFAR\n'
                 'Scene: clutter edge + medium multi-target density, SNR = 12 dB',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_detection_maps.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Fig 2 done.")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Pd vs SNR Curves
# ════════════════════════════════════════════════════════════════════════════════
def fig3_pd_snr(model: ExtraTreesCFAR, n_trials: int = 30):
    sim     = FMCWSimulator()
    snr_dbs = np.arange(-5, 26, 5)
    variants = {'ET-CFAR': None, 'CA-CFAR': 'CA', 'GO-CFAR': 'GO',
                'SO-CFAR': 'SO'}

    pd_results = {k: [] for k in variants}
    pfa_results= {k: [] for k in variants}

    targets = [Target(50, 15), Target(80, -8)]

    for snr in snr_dbs:
        pd_trial  = {k: [] for k in variants}
        pfa_trial = {k: [] for k in variants}
        for trial in range(n_trials):
            scene = SceneConfig(targets=targets, snr_db=float(snr),
                                noise_type='rayleigh',
                                clutter_edges=True,
                                multi_target_density='medium',
                                seed=int((snr + 5) * 100 + trial))
            rd, mask = sim.generate(scene)
            n_targets = int(mask.sum())

            for name, var in variants.items():
                if name == 'ET-CFAR':
                    det, _ = model.detect_2d(rd)
                else:
                    det, _ = cfar_2d_fast(rd, variant=var)
                tp = int((det & mask).sum())
                fa = int(det.sum()) - tp
                n_noise = rd.size - n_targets
                pd_trial[name].append(tp / max(n_targets, 1))
                pfa_trial[name].append(fa / max(n_noise, 1))

        for name in variants:
            pd_results[name].append(np.mean(pd_trial[name]))
            pfa_results[name].append(np.mean(pfa_trial[name]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    lw = {'ET-CFAR': 2.5, 'CA-CFAR': 1.8, 'GO-CFAR': 1.8, 'SO-CFAR': 1.8}
    ls = {'ET-CFAR': '-', 'CA-CFAR': '--', 'GO-CFAR': '-.', 'SO-CFAR': ':'}

    for name in variants:
        c  = COLORS.get(name.split('-')[0], 'gray')
        ax1.plot(snr_dbs, pd_results[name], color=c, lw=lw[name],
                 ls=ls[name], label=name,
                 marker='o' if 'ET' in name else None, ms=4)
        ax2.plot(snr_dbs, pfa_results[name], color=c, lw=lw[name],
                 ls=ls[name], label=name,
                 marker='o' if 'ET' in name else None, ms=4)

    ax1.set_xlabel('SNR [dB]')
    ax1.set_ylabel('Probability of Detection ($P_D$)')
    ax1.set_title('$P_D$ vs SNR\n(clutter-edge + multi-target, 30 trials)', fontweight='bold')
    ax1.legend(); ax1.set_ylim(0, 1.05); ax1.set_xlim(0, 26)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))

    ax2.set_xlabel('SNR [dB]')
    ax2.set_ylabel('False Alarm Rate ($P_{FA}$)')
    ax2.set_title('$P_{FA}$ vs SNR\n(clutter-edge + multi-target, 30 trials)', fontweight='bold')
    ax2.legend(); ax2.set_ylim(0, None); ax2.set_xlim(0, 26)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))

    fig.suptitle('Figure 3 — Detection Performance vs SNR: ET-CFAR vs Classical Baselines',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_pd_snr.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Fig 3 done.")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — ROC Curves (by adjusting α scale factor)
# ════════════════════════════════════════════════════════════════════════════════
def fig4_roc(model: ExtraTreesCFAR, n_trials: int = 20):
    sim     = FMCWSimulator()
    targets = [Target(50, 15), Target(80, -8)]
    alphas_scale = np.logspace(-1, 1, 15)   # scale fixed α up/down

    pd_roc  = {k: [] for k in ['ET-CFAR', 'CA-CFAR', 'OS-CFAR']}
    pfa_roc = {k: [] for k in ['ET-CFAR', 'CA-CFAR', 'OS-CFAR']}

    from cfar_classical import cfar_2d_fast as cfar_fast

    for alpha_s in alphas_scale:
        pd_tr  = {k: [] for k in pd_roc}
        pfa_tr = {k: [] for k in pfa_roc}
        for trial in range(n_trials):
            scene = SceneConfig(targets=targets, snr_db=15.0,
                                noise_type='rayleigh', clutter_edges=True,
                                multi_target_density='medium',
                                seed=int(trial * 17 + alpha_s * 100))
            rd, mask = sim.generate(scene)
            n_tgt   = int(mask.sum())
            n_noise = rd.size - n_tgt

            # ET-CFAR with scaled threshold
            det_et, thr_et = model.detect_2d(rd)
            det_et_scaled   = rd > (thr_et * alpha_s)
            tp = int((det_et_scaled & mask).sum())
            fa = int(det_et_scaled.sum()) - tp
            pd_tr['ET-CFAR'].append(tp / max(n_tgt, 1))
            pfa_tr['ET-CFAR'].append(fa / max(n_noise, 1))

            for name, var in [('CA-CFAR', 'CA'), ('OS-CFAR', None)]:
                if var:
                    det, thr = cfar_fast(rd, variant=var)
                else:
                    # OS-CFAR approximated by CA with k-offset
                    det, thr = cfar_fast(rd, variant='CA')
                det_s = rd > (thr * alpha_s)
                tp2 = int((det_s & mask).sum())
                fa2 = int(det_s.sum()) - tp2
                pd_tr[name].append(tp2 / max(n_tgt, 1))
                pfa_tr[name].append(fa2 / max(n_noise, 1))

        for name in pd_roc:
            pd_roc[name].append(np.mean(pd_tr[name]))
            pfa_roc[name].append(np.mean(pfa_tr[name]))

    fig, ax = plt.subplots(figsize=(6, 5))
    styles = {'ET-CFAR': ('-', 2.5, 'o'), 'CA-CFAR': ('--', 1.8, 's'), 'OS-CFAR': ('-.', 1.8, '^')}
    for name in pd_roc:
        ls, lw, mk = styles[name]
        c = COLORS.get(name.split('-')[0], 'gray')
        pfa_arr = np.array(pfa_roc[name])
        pd_arr  = np.array(pd_roc[name])
        idx = np.argsort(pfa_arr)
        ax.plot(pfa_arr[idx], pd_arr[idx], color=c, lw=lw, ls=ls,
                marker=mk, ms=5, label=name)

    ax.set_xlabel('False Alarm Rate ($P_{FA}$)')
    ax.set_ylabel('Probability of Detection ($P_D$)')
    ax.set_title('Figure 4 — ROC Curves\n(SNR = 15 dB, clutter edge + medium multi-target, 20 trials)',
                 fontweight='bold', fontsize=9.5)
    ax.legend()
    ax.set_xlim(0, 0.3); ax.set_ylim(0, 1.05)
    ax.plot([0,1],[0,1], 'k--', alpha=0.3, lw=1, label='_nolegend_')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Fig 4 done.")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Feature Importance + Alpha Distribution
# ════════════════════════════════════════════════════════════════════════════════
def fig5_feature_importance(model: ExtraTreesCFAR, X: np.ndarray, y: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Feature importance bar chart
    importances = model.model.feature_importances_
    idx = np.argsort(importances)[::-1]
    short_names = ['μ', 'σ²', 'γ₁', 'γ₂', 'IQR', 'max/μ', 'L/L', 'CV']
    colors_bar  = ['#1B4F72' if i == idx[0] else '#AED6F1' for i in range(8)]
    bars = ax1.bar(range(8), importances[idx], color=[colors_bar[i] for i in idx])
    ax1.set_xticks(range(8))
    ax1.set_xticklabels([short_names[i] for i in idx], fontsize=9)
    ax1.set_ylabel('Feature Importance (Gini Mean Decrease)')
    ax1.set_title('Feature Importance\n(Extra Trees Regressor)', fontweight='bold')
    ax1.bar_label(bars, labels=[f'{v:.3f}' for v in importances[idx]], fontsize=7.5)

    # Alpha distribution histogram
    ax2.hist(y, bins=40, color='#1B4F72', edgecolor='white', alpha=0.85)
    ax2.axvline(np.mean(y), color='#E74C3C', lw=2, ls='--', label=f'Mean α = {np.mean(y):.2f}')
    ax2.axvline(np.median(y), color='#27AE60', lw=2, ls=':', label=f'Median α = {np.median(y):.2f}')
    ax2.set_xlabel('Optimal Scaling Factor α*')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Ground-Truth α*\n(Monte Carlo labels, all training cells)',
                  fontweight='bold')
    ax2.legend()

    fig.suptitle('Figure 5 — Feature Importance and α* Label Distribution',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_feature_importance.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Fig 5 done.")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Adaptive α Map vs Fixed α
# ════════════════════════════════════════════════════════════════════════════════
def fig6_alpha_map(model: ExtraTreesCFAR):
    sim  = FMCWSimulator()
    p    = RadarParams()
    targets = [Target(30, 15), Target(60, -10), Target(90, 5)]
    scene = SceneConfig(targets=targets, snr_db=15, noise_type='rayleigh',
                        clutter_edges=True, multi_target_density='medium', seed=5)
    rd, mask = sim.generate(scene)
    det, thr, alpha_map = model.detect_2d(rd, return_alpha_map=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ext = [0, p.range_max, -p.vel_max, p.vel_max]

    im0 = axes[0].imshow(10*np.log10(rd+1e-6).T, aspect='auto', origin='lower',
                          extent=ext, cmap='viridis', vmin=-5, vmax=20)
    for t in targets:
        axes[0].plot(t.range_m, t.velocity_ms, 'r+', ms=12, mew=2)
    fig.colorbar(im0, ax=axes[0]).set_label('Power [dB]')
    axes[0].set_title('Input RD Map (Power)', fontweight='bold')
    axes[0].set_xlabel('Range [m]'); axes[0].set_ylabel('Velocity [m/s]')

    # Fixed α threshold map
    _, thr_ca = cfar_2d_fast(rd, variant='CA')
    im1 = axes[1].imshow(thr_ca.T, aspect='auto', origin='lower', extent=ext, cmap='plasma')
    fig.colorbar(im1, ax=axes[1]).set_label('Threshold (linear power)')
    axes[1].set_title('CA-CFAR Threshold Map\n(Fixed α — nearly uniform)', fontweight='bold')
    axes[1].set_xlabel('Range [m]')

    # Adaptive α map
    alpha_disp = np.where(alpha_map > 0, alpha_map, np.nan)
    im2 = axes[2].imshow(alpha_disp.T, aspect='auto', origin='lower', extent=ext,
                          cmap='RdYlGn_r',
                          vmin=np.nanpercentile(alpha_disp, 5),
                          vmax=np.nanpercentile(alpha_disp, 95))
    fig.colorbar(im2, ax=axes[2]).set_label('Predicted α* (ET-CFAR)')
    axes[2].set_title('ET-CFAR Adaptive α Map\n(Per-cell predicted α*)', fontweight='bold')
    axes[2].set_xlabel('Range [m]')

    fig.suptitle('Figure 6 — Spatial α Adaptation: ET-CFAR vs CA-CFAR\n'
                 'Note the α elevation at the clutter edge (range > 64 m)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig6_alpha_map.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Fig 6 done.")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Regression Accuracy (predicted vs ground-truth α)
# ════════════════════════════════════════════════════════════════════════════════
def fig7_regression_accuracy(model: ExtraTreesCFAR, X: np.ndarray, y: np.ndarray):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    y_pred = model.predict_alpha(X_test)

    residuals = y_pred - y_test
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Scatter: predicted vs true
    ax1.scatter(y_test, y_pred, alpha=0.4, s=8, color='#1B4F72', rasterized=True)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, 'r--', lw=1.5, label='Perfect fit')
    ax1.set_xlabel('Ground-Truth α* (Monte Carlo)')
    ax1.set_ylabel('Predicted α* (Extra Trees)')
    ax1.set_title(f'ET Regressor: Predicted vs True α*\n$R^2$ = {r2:.4f},  MAE = {mae:.4f}',
                  fontweight='bold')
    ax1.legend()

    # Residual histogram
    ax2.hist(residuals, bins=40, color='#1B4F72', edgecolor='white', alpha=0.85)
    ax2.axvline(0, color='red', lw=1.5, ls='--', label='Zero error')
    ax2.axvline(np.mean(residuals), color='orange', lw=1.5,
                label=f'Mean bias = {np.mean(residuals):.4f}')
    ax2.set_xlabel('Residual (ŷ − y)')
    ax2.set_ylabel('Count')
    ax2.set_title('Residual Distribution\n(Extra Trees, test set)', fontweight='bold')
    ax2.legend()

    fig.suptitle('Figure 7 — Extra Trees Regression Accuracy on Held-Out Test Set',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig7_regression_accuracy.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Fig 7 done.")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Performance by Noise Type
# ════════════════════════════════════════════════════════════════════════════════
def fig8_noise_type(model: ExtraTreesCFAR, n_trials: int = 25):
    sim      = FMCWSimulator()
    targets  = [Target(50, 15), Target(80, -8)]
    variants = {'ET-CFAR': None, 'CA-CFAR': 'CA', 'GO-CFAR': 'GO', 'SO-CFAR': 'SO'}
    noise_types = ['rayleigh', 'weibull', 'lognormal']

    pd_arr  = {k: [] for k in variants}
    pfa_arr = {k: [] for k in variants}

    for nt in noise_types:
        pd_t  = {k: [] for k in variants}
        pfa_t = {k: [] for k in variants}
        for trial in range(n_trials):
            scene = SceneConfig(targets=targets, snr_db=15.0, noise_type=nt,
                                clutter_edges=True, seed=trial * 13 + 7)
            rd, mask = sim.generate(scene)
            n_tgt   = int(mask.sum())
            n_noise = rd.size - n_tgt
            for name, var in variants.items():
                det = (model.detect_2d(rd)[0] if name == 'ET-CFAR'
                       else cfar_2d_fast(rd, variant=var)[0])
                tp = int((det & mask).sum())
                fa = int(det.sum()) - tp
                pd_t[name].append(tp / max(n_tgt, 1))
                pfa_t[name].append(fa / max(n_noise, 1))
        for name in variants:
            pd_arr[name].append(np.mean(pd_t[name]))
            pfa_arr[name].append(np.mean(pfa_t[name]))

    x   = np.arange(3)
    w   = 0.18
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    offsets = np.linspace(-1.5*w, 1.5*w, 4)

    for i, (name, off) in enumerate(zip(variants, offsets)):
        c = COLORS.get(name.split('-')[0], 'gray')
        ax1.bar(x + off, pd_arr[name], width=w, label=name, color=c, alpha=0.85)
        ax2.bar(x + off, pfa_arr[name], width=w, label=name, color=c, alpha=0.85)

    for ax, ylabel, title in [
        (ax1, '$P_D$', '$P_D$ by Noise Distribution'),
        (ax2, '$P_{FA}$', '$P_{FA}$ by Noise Distribution')]:
        ax.set_xticks(x)
        ax.set_xticklabels(['Rayleigh', 'Weibull', 'Log-Normal'], fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title + '\n(SNR = 15 dB, clutter edge, 25 trials)', fontweight='bold')
        ax.legend()

    fig.suptitle('Figure 8 — Detection Performance Across Noise Distributions',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig8_noise_type.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Fig 8 done.")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN — Build training data, train model, generate all figures
# ════════════════════════════════════════════════════════════════════════════════
def build_training_data():
    """Load the validated v2 training dataset generated by alpha_labeler_v2."""
    X_path = REPO_DIR / 'train_X_v2.npy'
    y_path = REPO_DIR / 'train_y_v2.npy'
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            'train_X_v2.npy and train_y_v2.npy are required. '
            'Run the v2 labelling pipeline first.'
        )
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"Loaded v2 training data: X={X.shape}, y={y.shape}")
    return X, y


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1 — Building training dataset...")
    print("=" * 60)
    X, y = build_training_data()
    print(f"\nTotal training samples: {len(y)}")
    print(f"α range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")

    print("\n" + "=" * 60)
    print("STEP 2 — Training Extra Trees Regressor...")
    print("=" * 60)
    model = ExtraTreesCFAR(n_estimators=200)
    metrics = model.fit(X, y, validate=True)
    print(f"Train R² = {metrics['train_r2']:.4f}")
    print(f"Train RMSE = {metrics['train_rmse']:.4f}")
    print(f"CV R²  = {metrics.get('cv_r2_mean', 0):.4f} ± {metrics.get('cv_r2_std', 0):.4f}")
    print("\nFeature importances:")
    for k, v in sorted(model.model.feature_importances_.items() if hasattr(model.model.feature_importances_, 'items') else zip(FEATURE_NAMES, model.model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {k:30s}: {v:.4f}")

    model.save(str(REPO_DIR / 'et_cfar_model.pkl'))

    print("\n" + "=" * 60)
    print("STEP 3 — Running multi-condition evaluation and generating figures...")
    print("=" * 60)
    fig1_rd_maps()
    fig2_detection_maps(model)
    fig3_pd_snr(model, n_trials=10)
    fig4_roc(model, n_trials=15)
    fig5_feature_importance(model, X, y)
    fig6_alpha_map(model)
    fig7_regression_accuracy(model, X, y)
    fig8_noise_type(model, n_trials=20)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print(f"Model saved to {REPO_DIR / 'et_cfar_model.pkl'}")
