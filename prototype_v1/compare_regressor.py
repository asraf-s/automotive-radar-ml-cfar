#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:51:01 2026

@author: asrafs
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# =====================================================
# LOAD DATASET
# =====================================================
df = pd.read_csv("radar_continuous_f1_dataset.csv")

X = df[[
    "Mean", "Std", "Max", "Min",
    "PTP", "Median",
    "Sigma", "NumTargets"
]]

y = df["BestThreshold"]

# =====================================================
# SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MODELS
# =====================================================
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        random_state=42
    ),

    "Extra Trees Regressor": ExtraTreesRegressor(
        n_estimators=500,
        max_depth=14,
        random_state=42
    )
}

# =====================================================
# TRAIN + EVALUATE
# =====================================================
for name, model in models.items():

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    print(f"\n{name}")
    print("MAE:", round(mae,3))
    print("R2 :", round(r2,3))