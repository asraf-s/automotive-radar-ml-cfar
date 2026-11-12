import numpy as np
import time
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURE_NAMES = [
    "cell_db", "mean_db", "std_db", "skew", "kurt",
    "median_db", "min_db", "max_db", "mad_db", "iqr_db",
    "p10_db", "p90_db", "sum_db", "sum_lin", "n_cells",
    "row", "col", "row_norm", "col_norm"
]


class ExtraTreesCFAR:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
        scaler: Optional[StandardScaler] = None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = scaler or StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, validate: bool = False) -> Dict[str, float]:
        """Fit the ExtraTrees regressor on provided features and labels."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        y_pred = self.model.predict(X_scaled)
        metrics = {
            "train_rmse": mean_squared_error(y, y_pred, squared=False),
            "train_mae": mean_absolute_error(y, y_pred),
            "train_r2": r2_score(y, y_pred),
        }
        if validate:
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_r2 = cross_val_score(self.model, X_scaled, y, cv=kf, scoring="r2")
            cv_rmse = -cross_val_score(self.model, X_scaled, y, cv=kf, scoring="neg_root_mean_squared_error")
            metrics["cv_r2_mean"] = np.mean(cv_r2)
            metrics["cv_r2_std"] = np.std(cv_r2)
            metrics["cv_rmse_mean"] = np.mean(cv_rmse)
        return metrics

    def predict_alpha(self, X: np.ndarray) -> np.ndarray:
        """Predict threshold scaling factor alpha for each feature vector."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def detect_2d(self, Z_arr: np.ndarray, return_alpha_map: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Apply ET-CFAR using the same window and alpha definition as v2 labels.

        Feature extraction and Extra-Trees prediction are batched so that the
        trained model is called once for the complete set of valid CUT cells.
        The CFAR geometry and threshold definition are unchanged.
        """
        from alpha_labeler_v2 import extract_features

        rd = np.asarray(Z_arr, dtype=np.float32)
        Nr, Nd = rd.shape
        guard = 2
        train = 4
        W_total = guard + train + 1
        W_guard = guard + 1

        detections = np.zeros((Nr, Nd), dtype=bool)
        threshold_map = np.zeros((Nr, Nd), dtype=np.float32)
        alpha_map = np.zeros((Nr, Nd), dtype=np.float32)

        features_list = []
        ref_means = []
        coords = []

        for ri in range(W_total, Nr - W_total):
            for di in range(Nd):
                Z = []
                for dr in range(-W_total, W_total + 1):
                    for dd in range(-W_total, W_total + 1):
                        d = max(abs(dr), abs(dd))
                        if W_guard <= d <= W_total:
                            r2 = ri + dr
                            d2 = (di + dd) % Nd
                            if 0 <= r2 < Nr:
                                Z.append(rd[r2, d2])

                Z_arr_local = np.asarray(Z, dtype=np.float32)
                features_list.append(extract_features(Z_arr_local))
                ref_means.append(float(np.mean(Z_arr_local)))
                coords.append((ri, di))

        if not features_list:
            if return_alpha_map:
                return detections, threshold_map, alpha_map
            return detections, threshold_map

        features = np.asarray(features_list, dtype=np.float32)
        alpha_pred = np.maximum(self.predict_alpha(features), 0.0).astype(np.float32)
        ref_means = np.asarray(ref_means, dtype=np.float32)

        thresholds = alpha_pred * ref_means
        coords = np.asarray(coords, dtype=np.int32)
        rows = coords[:, 0]
        cols = coords[:, 1]

        threshold_map[rows, cols] = thresholds
        alpha_map[rows, cols] = alpha_pred
        detections[rows, cols] = rd[rows, cols] > thresholds

        if return_alpha_map:
            return detections, threshold_map, alpha_map
        return detections, threshold_map

    def benchmark_inference(self, X: np.ndarray, n_repeats: int = 10) -> float:
        """Benchmark average inference time per sample."""
        X_scaled = self.scaler.transform(X)
        start = time.perf_counter()
        for _ in range(n_repeats):
            _ = self.model.predict(X_scaled)
        elapsed = time.perf_counter() - start
        return elapsed / (n_repeats * len(X))

    def save(self, path: str):
        """Save model and scaler to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
            }, f)

    @classmethod
    def load(cls, path: str) -> "ExtraTreesCFAR":
        """Load model and scaler from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            n_estimators=data["n_estimators"],
            max_depth=data["max_depth"],
            random_state=data["random_state"]
        )
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.is_fitted = True
        return obj

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from fmcw_simulator import FMCWSimulator, SceneConfig, Target
    from alpha_labeler_v2 import label_rd_map

    repo_dir = Path(__file__).resolve().parent
    X_path = repo_dir / 'train_X_v2.npy'
    y_path = repo_dir / 'train_y_v2.npy'

    if X_path.exists() and y_path.exists():
        X = np.load(X_path)
        y = np.load(y_path)
        print(f"Loaded v2 training data: X={X.shape}, y={y.shape}")
    else:
        sim = FMCWSimulator()
        scene = SceneConfig(
            targets=[Target(50, 15)],
            snr_db=15,
            noise_type='rayleigh',
            clutter_edges=True,
            seed=42
        )
        rd, mask = sim.generate(scene)
        print("Generating v2 alpha labels...")
        _, X, y = label_rd_map(rd, subsample=4, n_mc=500)
        np.save(X_path, X)
        np.save(y_path, y)
        print(f"Saved v2 training data: X={X.shape}, y={y.shape}")

    print(f"Feature count: {X.shape[1]}")
    print(f"Alpha range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Alpha mean/std: {y.mean():.3f} / {y.std():.3f}")

    model = ExtraTreesCFAR(n_estimators=100, random_state=42)
    metrics = model.fit(X, y, validate=True)

    print(f"Train RMSE={metrics['train_rmse']:.4f}")
    print(f"Train MAE={metrics['train_mae']:.4f}")
    print(f"Train R2={metrics['train_r2']:.4f}")
    if 'cv_r2_mean' in metrics:
        print(f"5-fold CV R2={metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        print(f"5-fold CV RMSE={metrics['cv_rmse_mean']:.4f}")

    sim = FMCWSimulator()
    scene = SceneConfig(
        targets=[Target(50, 15)],
        snr_db=15,
        noise_type='rayleigh',
        clutter_edges=True,
        seed=42
    )
    rd, mask = sim.generate(scene)

    det, thr, alpha_map = model.detect_2d(rd, return_alpha_map=True)

    tp_mask = det & mask
    fp_mask = det & ~mask
    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    print(f"ET-CFAR: detections={int(det.sum())}, true positives={tp}, false positives={fp}")

    range_axis = np.arange(rd.shape[0]) * sim.p.range_res
    velocity_axis = (np.arange(rd.shape[1]) - rd.shape[1] // 2) * sim.p.vel_res
    rd_shifted = np.fft.fftshift(rd, axes=1)
    det_shifted = np.fft.fftshift(det, axes=1)
    mask_shifted = np.fft.fftshift(mask, axes=1)
    tp_shifted = np.fft.fftshift(tp_mask, axes=1)
    fp_shifted = np.fft.fftshift(fp_mask, axes=1)

    rd_db = 10 * np.log10(rd_shifted + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        rd_db.T,
        extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
        aspect='auto',
        origin='lower'
    )

    fp_r, fp_d = np.where(fp_shifted)
    tp_r, tp_d = np.where(tp_shifted)
    true_r, true_d = np.where(mask_shifted)

    if len(fp_r):
        plt.scatter(range_axis[fp_r], velocity_axis[fp_d], marker='o', s=55, facecolors='none', linewidths=1.5, label='False alarm')
    if len(tp_r):
        plt.scatter(range_axis[tp_r], velocity_axis[tp_d], marker='o', s=80, facecolors='none', linewidths=2, label='True detection')
    plt.scatter(range_axis[true_r], velocity_axis[true_d], marker='x', s=120, linewidths=2, label='Ground truth')

    plt.xlabel('Range (m)')
    plt.ylabel('Radial Velocity (m/s)')
    plt.title('ET-CFAR Detection on FMCW Range-Doppler Map')
    plt.colorbar(label='Power (dB)')
    plt.legend()
    plt.tight_layout()

    valid_alpha = alpha_map[alpha_map > 0]
    plt.figure(figsize=(9, 5))
    plt.hist(valid_alpha.ravel(), bins=30)
    plt.xlabel('Predicted α')
    plt.ylabel('Number of cells')
    plt.title('ET-CFAR Predicted Threshold Scaling Factor Distribution')
    plt.tight_layout()

    plt.show()
