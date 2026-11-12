#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:04:37 2026

@author: asrafs
"""

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

# =====================================================
# 1. TRAIN FINAL MODEL
# =====================================================
df = pd.read_csv("radar_continuous_f1_dataset.csv")

X = df.drop(columns=["BestThreshold"])
y = df["BestThreshold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=14,
    random_state=42
)
model.fit(X_train, y_train)

# =====================================================
# 2. RADAR PARAMETERS
# =====================================================
c = 3e8
fc = 77e9
B = 150e6
T_chirp = 40e-6
slope = B / T_chirp

Nr = 256
Nd = 64

t_chirp = np.linspace(0, T_chirp, Nr)

range_res = c / (2 * B)
range_axis = np.arange(Nr//2) * range_res
vel_axis = np.linspace(-20, 20, Nd)

Tr, Td = 6, 4
Gr, Gd = 2, 1

# =====================================================
# METRICS
# =====================================================
correct = 0
missed = 0
false = 0

range_errors = []
vel_errors = []
times = []

# =====================================================
# 3. TEST MANY FRAMES
# =====================================================
for frame in range(100):

    start = time.time()

    num_targets = np.random.randint(1, 4)
    true_ranges = np.random.uniform(20, 150, num_targets)
    true_vels = np.random.uniform(-15, 15, num_targets)

    sigma = np.random.uniform(0.5, 4.0)

    Mix = np.zeros((Nd, Nr))

    for d in range(Nd):

        tt = t_chirp
        Tx = np.cos(2*np.pi*(fc*tt + (slope*tt**2)/2))

        Rx = np.zeros(Nr)

        for R, v in zip(true_ranges, true_vels):

            R_now = R + v*d*T_chirp
            td = 2*R_now/c
            amp = np.random.uniform(0.3, 1.0)

            echo = amp*np.cos(
                2*np.pi*(fc*(tt-td)+(slope*(tt-td)**2)/2)
            )

            Rx += echo

        Mix[d,:] = Tx * Rx

    Mix += np.random.normal(0, sigma, Mix.shape)

    # FFT
    rd = np.fft.fft(Mix, axis=1)
    rd = rd[:, :Nr//2]
    rd = np.fft.fftshift(np.fft.fft(rd, axis=0), axes=0)

    rd_map = np.abs(rd)
    rd_map = rd_map / (np.max(rd_map)+1e-12)

    # Features
    signal = np.mean(rd_map, axis=0)

    feat = pd.DataFrame([{
        "Mean": np.mean(signal),
        "Std": np.std(signal),
        "Max": np.max(signal),
        "Min": np.min(signal),
        "PTP": np.ptp(signal),
        "Median": np.median(signal),
        "Sigma": sigma,
        "NumTargets": num_targets
    }])

    offset = model.predict(feat)[0]
    offset = max(2.0, min(offset, 4.5))

    # =================================================
    # 2D CA-CFAR
    # =================================================
    det = np.zeros_like(rd_map)

    for r in range(Td+Gd, Nd-(Td+Gd)):
        for cbin in range(Tr+Gr, (Nr//2)-(Tr+Gr)):

            win = rd_map[
                r-(Td+Gd):r+(Td+Gd)+1,
                cbin-(Tr+Gr):cbin+(Tr+Gr)+1
            ]

            mask = np.ones(win.shape, dtype=bool)

            mask[Td:Td+2*Gd+1,
                 Tr:Tr+2*Gr+1] = False

            noise = np.mean(win[mask])
            th = noise * offset

            if rd_map[r,cbin] > th:
                det[r,cbin] = rd_map[r,cbin]

    # Peak Filter
    final_det = []
    idx = np.argwhere(det > 0)

    for rr, cc in idx:

        local = det[
            max(0,rr-1):min(Nd,rr+2),
            max(0,cc-1):min(Nr//2,cc+2)
        ]

        if det[rr,cc] >= np.max(local):
            final_det.append((rr,cc))

    # =================================================
    # MATCHING
    # =================================================
    matched = set()

    for tr, tv in zip(true_ranges, true_vels):

        found = False

        for k, (rr, cc) in enumerate(final_det):

            dr = range_axis[cc]
            dv = vel_axis[rr]

            if abs(dr-tr) < 3 and abs(dv-tv) < 3 and k not in matched:
                correct += 1
                range_errors.append(abs(dr-tr))
                vel_errors.append(abs(dv-tv))
                matched.add(k)
                found = True
                break

        if not found:
            missed += 1

    false += max(0, len(final_det) - len(matched))

    times.append((time.time()-start)*1000)

# =====================================================
# FINAL METRICS
# =====================================================
precision = correct / (correct + false + 1e-12)
recall = correct / (correct + missed + 1e-12)
f1 = 2 * precision * recall / (precision + recall + 1e-12)

print("Correct Detections :", correct)
print("Missed Targets     :", missed)
print("False Alarms       :", false)

print("\nPrecision :", round(precision,3))
print("Recall    :", round(recall,3))
print("F1 Score  :", round(f1,3))

print("\nAvg Range Error (m):",
      round(np.mean(range_errors),3))

print("Avg Vel Error (m/s):",
      round(np.mean(vel_errors),3))

print("Avg Time (ms/frame):",
      round(np.mean(times),3))