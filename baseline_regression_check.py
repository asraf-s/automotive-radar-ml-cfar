"""
ET-CFAR Baseline Regression Evaluation
======================================

Purpose:
Evaluate a newly trained Extra Trees regression model on held-out data.

Dataset:
    train_X_v2.npy
    train_y_v2.npy

This script:
    1. Loads the existing dataset.
    2. Splits the data into training and test sets.
    3. Fits preprocessing on training data only.
    4. Trains a fresh Extra Trees model.
    5. Evaluates predictions on held-out samples.
    6. Reports regression metrics and feature importances.

IMPORTANT:
This script does not load, modify, or overwrite the existing
et_cfar_model.pkl file.
"""

from pathlib import Path

import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PROJECT PATHS
# ============================================================

REPO_DIR = Path(__file__).resolve().parent

X_PATH = REPO_DIR / "train_X_v2.npy"
Y_PATH = REPO_DIR / "train_y_v2.npy"


# ============================================================
# 2. MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("ET-CFAR BASELINE REGRESSION EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load existing dataset
    # --------------------------------------------------------

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print("\nDataset information")
    print("-" * 40)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if X.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional.")

    if y.ndim != 1:
        raise ValueError("Target vector must be one-dimensional.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Feature and target sample counts do not match."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "Target vector contains NaN or infinite values."
        )

    if X.shape[1] != 8:
        raise ValueError(
            f"Expected 8 input features, found {X.shape[1]}."
        )

    # --------------------------------------------------------
    # Split into training and test datasets
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    print("\nDataset split")
    print("-" * 40)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")

    # --------------------------------------------------------
    # Preprocessing
    # Fit scaler on training data only
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # Train a NEW Extra Trees model
    # --------------------------------------------------------

    print("\nTraining a fresh Extra Trees model...")

    model = ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)

    # --------------------------------------------------------
    # Predict on held-out test data
    # --------------------------------------------------------

    y_pred = model.predict(X_test_scaled)

    # --------------------------------------------------------
    # Calculate regression metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    # Simple baseline: predict the training-target mean
    mean_predictions = np.full(
        shape=len(y_test),
        fill_value=np.mean(y_train),
        dtype=float,
    )

    baseline_mae = mean_absolute_error(
        y_test,
        mean_predictions,
    )

    # --------------------------------------------------------
    # Display evaluation results
    # --------------------------------------------------------

    print("\nHeld-out regression results")
    print("=" * 60)

    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.6f}")

    print(
        f"Mean-predictor baseline MAE: "
        f"{baseline_mae:.6f}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print("\nFeature importances")
    print("-" * 40)

    for index, importance in enumerate(
        model.feature_importances_
    ):
        print(
            f"Feature {index}: {importance:.6f}"
        )

    # --------------------------------------------------------
    # Safety and interpretation notes
    # --------------------------------------------------------

    print("\nEvaluation notes")
    print("-" * 40)

    print(
        "The existing et_cfar_model.pkl was not modified."
    )

    print(
        "The scaler was fitted using training data only."
    )

    print(
        "This evaluation uses a random held-out split."
    )

    print(
        "Generalization to independent radar scenes and "
        "unseen noise distributions remains to be tested."
    )

    print("\nEvaluation completed successfully.")


# ============================================================
# 3. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()