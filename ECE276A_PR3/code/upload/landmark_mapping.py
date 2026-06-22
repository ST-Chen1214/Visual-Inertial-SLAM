"""
ECE 276A Project 3 — Part 3
Landmark Mapping via EKF Update (fixed IMU trajectory assumed)
"""

import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import os

from pr3_utils import (project_stereo, jacobian_landmark, triangulate,
                        load_dataset, plot_trajectory_and_landmarks,
                        inv_SE3)
from imu_localization import imu_ekf_prediction


# ============================================================
#  Helper: select which features to use
# ============================================================

def select_features(features, max_features=500, min_observations=3):
    """
    Select a manageable subset of feature indices.

    Strategy:
      1. Count how many times each feature is observed.
      2. Filter out rarely-observed features.
      3. Randomly subsample if still too many.

    Returns: array of selected feature indices (M_sel,)
    """
    M = features.shape[1]
    N = features.shape[2]

    obs_count = np.sum(~np.any(features == -1, axis=0), axis=1)  # (M,)
    good = np.where(obs_count >= min_observations)[0]
    print(f"  Features with ≥{min_observations} obs: {len(good)} / {M}")

    if len(good) > max_features:
        good = np.random.choice(good, max_features, replace=False)
        good.sort()
    print(f"  Using {len(good)} features")
    return good


# ============================================================
#  Core EKF Landmark Mapping
# ============================================================

def landmark_mapping_ekf(T_hist, features, iTC_l, iTC_r, K_l, K_r,
                          V_noise=4.0, init_sigma=1.0,
                          max_features=500, time_subsample=1,
                          min_observations=3):
    """
    EKF update-only for landmark positions (Part 3).

    The IMU trajectory T_hist is assumed correct (no pose update).
    Landmarks are modelled as static → no prediction step needed.

    Args:
        T_hist        : (N,4,4)  IMU pose history (world frame)
        features      : (4,M,N)  stereo feature pixel coords
        iTC_l / iTC_r : (4,4)    camera extrinsics
        K_l / K_r     : (3,3)    camera intrinsics
        V_noise       : scalar   observation noise variance (pixels²)
        init_sigma    : scalar   initial landmark position std-dev [m]
        max_features  : int      max landmarks to track
        time_subsample: int      only process every k-th timestep
        min_observations: int    minimum times a feature must appear

    Returns:
        mu_m   : (3, M_sel)  estimated landmark positions (world frame)
        init   : (M_sel,)    boolean — whether each landmark is initialised
        sel_idx: (M_sel,)    original feature indices that were selected
    """
    N = T_hist.shape[0]
    M_all = features.shape[1]

    V = np.eye(4) * V_noise

    # Select feature subset
    sel_idx = select_features(features, max_features, min_observations)
    M = len(sel_idx)

    # Initialise state: block-diagonal covariance for efficiency
    mu_m   = np.zeros((3, M))
    Sigma  = np.eye(3 * M) * (init_sigma ** 2)   # block diagonal
    init_f = np.zeros(M, dtype=bool)

    print(f"\n[Landmark EKF] N={N}, M_sel={M}, subsample={time_subsample}")

    for t in range(0, N, time_subsample):
        T  = T_hist[t]
        Ti = inv_SE3(T)

        # Extract observations for selected features
        zt = features[:, sel_idx, t]   # (4, M)
        obs_mask = ~np.any(zt == -1, axis=0)   # (M,)
        obs_idx  = np.where(obs_mask)[0]

        if len(obs_idx) == 0:
            continue

        # ---- initialise new landmarks ----
        for k in obs_idx:
            if init_f[k]:
                continue
            m = triangulate(zt[:, k], T, iTC_l, iTC_r, K_l, K_r)
            if m is None:
                continue
            # Extra: reject if too far from current robot position
            dist = np.linalg.norm(m[:2] - T[:2, 3])
            if dist > 80.0:
                continue
            mu_m[:, k] = m
            init_f[k]  = True

        # ---- EKF update for all initialised landmarks ----
        valid = obs_idx[init_f[obs_idx]]

        for k in valid:
            m_k   = mu_m[:, k]
            zpred = project_stereo(m_k, T, iTC_l, iTC_r, K_l, K_r)
            if zpred is None:
                continue

            H    = jacobian_landmark(m_k, T, iTC_l, iTC_r, K_l, K_r)   # 4×3
            sk   = slice(3 * k, 3 * k + 3)
            Sigk = Sigma[sk, sk]                                           # 3×3

            # Innovation covariance
            S    = H @ Sigk @ H.T + V                                     # 4×4
            Kgain = Sigk @ H.T @ np.linalg.inv(S)                        # 3×4

            # Update
            innov     = zt[:, k] - zpred
            mu_m[:, k]  += Kgain @ innov
            Sigma[sk, sk] = (np.eye(3) - Kgain @ H) @ Sigk

        if t % 200 == 0:
            print(f"  t={t}/{N}  initialised={init_f.sum()}/{M}")

    return mu_m, init_f, sel_idx


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\ucsd\handouts\ece276a\ECE276A_PR3\ECE276A_PR3\data\ECE276A_P3_Data\dataset00"

    dname = os.path.basename(dataset_path)
    print(f"=== Part 3: Landmark Mapping ===")
    print(f"Dataset: {dataset_path}")

    # Load data
    data = load_dataset(dataset_path)
    t, v, w = data['t'], data['v'], data['w']
    features = data['features']
    K_l, K_r = data['K_l'], data['K_r']
    iTC_l, iTC_r = data['iTC_l'], data['iTC_r']

    # Part 1: get IMU trajectory
    print("\n[Running IMU EKF prediction...]")
    T_hist, Sigma_hist = imu_ekf_prediction(t, v, w)

    # Part 3: landmark mapping
    print("\n[Running Landmark Mapping EKF...]")
    mu_m, init_f, sel_idx = landmark_mapping_ekf(
        T_hist, features, iTC_l, iTC_r, K_l, K_r,
        V_noise=4.0,
        init_sigma=2.0,
        max_features=500,
        time_subsample=1,
        min_observations=3,
    )

    # ---- Visualise ----
    fig = plot_trajectory_and_landmarks(
        T_hist, mu_m, init_f,
        title=f'Part 3 — Landmark Mapping ({dname})',
        save_path=f'landmark_map_{dname}.png'
    )

    # ---- Save ----
    out_npz = f'landmark_result_{dname}.npz'
    np.savez(out_npz, mu_m=mu_m, init_f=init_f, sel_idx=sel_idx,
             T_hist=T_hist)
    print(f"Results saved: {out_npz}")