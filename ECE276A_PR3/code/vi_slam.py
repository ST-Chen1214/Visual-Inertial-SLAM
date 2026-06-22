"""
ECE 276A Project 3 — Part 4  (v2: numerically stable)
Visual-Inertial SLAM

Key improvements over v1:
  - Per-landmark 3×3 covariances (no 3M×3M matrix → stable & fast)
  - Innovation gating (chi-squared test, rejects outlier observations)
  - Pose update warmup (don't correct pose until IMU has moved)
  - Disparity + depth + reprojection filtering in triangulation (in pr3_utils)
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import os

from pr3_utils import (hat, curlyhat, inv_SE3,
                        project_stereo, jacobian_landmark, jacobian_pose,
                        triangulate, load_dataset, plot_trajectory_and_landmarks)
from landmark_mapping import select_features


def vi_slam(t, v, w, features, iTC_l, iTC_r, K_l, K_r,
            W_v=5e-5,
            W_w=1e-5,
            V_noise=25.0,
            max_features=400,
            n_update_per_step=15,
            time_subsample_init=3,
            min_observations=3,
            init_sigma=1.0,
            innov_gate=50.0,
            chi2_gate=15.87,
            warmup_steps=50):
    """
    Full VI-SLAM with per-landmark independent 3×3 covariances.

    State:
        T_t ∈ SE(3)   — IMU-in-world pose
        m_k ∈ R^3     — each landmark (independent, no inter-landmark coupling)

    Covariance blocks (per landmark k):
        Sig_TT      : (6,6)   pose covariance
        Sig_Tm[k]   : (6,3)   pose × landmark-k cross-term
        Sig_m[k]    : (3,3)   landmark-k covariance

    Returns:
        T_hist : (N,4,4)   estimated IMU pose history
        mu_m   : (3,M_sel) estimated landmark positions
        init_f : (M_sel,)  initialised flags
        sel_idx: (M_sel,)  original feature indices
    """
    N = len(t)
    W = np.diag([W_v]*3 + [W_w]*3)
    V = np.eye(4) * V_noise

    sel_idx = select_features(features, max_features, min_observations)
    M = len(sel_idx)
    print(f"[VI-SLAM] N={N}  M={M}  V_noise={V_noise}  warmup={warmup_steps}")

    # State
    T      = np.eye(4)
    Sig_TT = np.eye(6) * 1e-8
    mu_m   = np.zeros((3, M))
    Sig_m  = np.tile(np.eye(3) * init_sigma**2, (M, 1, 1))   # (M,3,3)
    Sig_Tm = np.zeros((M, 6, 3))                               # (M,6,3)
    init_f = np.zeros(M, dtype=bool)

    T_hist = np.zeros((N, 4, 4))
    T_hist[0] = T

    for i in range(1, N):
        tau = t[i] - t[i-1]
        u   = np.r_[v[:, i-1], w[:, i-1]]

        # ===== PREDICTION =====
        T      = T @ expm(tau * hat(u))
        F      = expm(-tau * curlyhat(u))
        Sig_TT = F @ Sig_TT @ F.T + W * tau
        # Propagate cross-terms: Σ_Tm_k <- F @ Σ_Tm_k
        if init_f.any():
            Sig_Tm[init_f] = np.einsum('ij,kjl->kil', F, Sig_Tm[init_f])

        # ===== COLLECT OBSERVATIONS =====
        zt       = features[:, sel_idx, i]
        obs_mask = ~np.any(zt == -1, axis=0)
        obs_idx  = np.where(obs_mask)[0]

        if len(obs_idx) == 0:
            T_hist[i] = T
            continue

        # Initialise new landmarks
        if i % time_subsample_init == 0:
            for k in obs_idx:
                if init_f[k]:
                    continue
                m = triangulate(zt[:, k], T, iTC_l, iTC_r, K_l, K_r)
                if m is None:
                    continue
                mu_m[:, k] = m
                init_f[k]  = True

        # Select landmarks to update
        valid = obs_idx[init_f[obs_idx]]
        if len(valid) == 0:
            T_hist[i] = T
            continue

        if len(valid) > n_update_per_step:
            valid = np.random.choice(valid, n_update_per_step, replace=False)

        # ===== SEQUENTIAL EKF UPDATE =====
        I6 = np.eye(6)
        I3 = np.eye(3)

        # Accumulate total pose correction across all features this timestep
        # Apply once at the end → avoids jerky per-feature jumps
        total_delta_xi = np.zeros(6)
        n_pose_updates = 0

        for k in valid:
            m_k   = mu_m[:, k]
            zpred = project_stereo(m_k, T, iTC_l, iTC_r, K_l, K_r)
            if zpred is None:
                continue

            innov = zt[:, k] - zpred

            # L∞ gating
            if np.max(np.abs(innov)) > innov_gate:
                continue

            HT = jacobian_pose(m_k, T, iTC_l, iTC_r, K_l, K_r)
            Hm = jacobian_landmark(m_k, T, iTC_l, iTC_r, K_l, K_r)

            # Innovation covariance
            STT = HT @ Sig_TT    @ HT.T
            STm = HT @ Sig_Tm[k] @ Hm.T
            Smm = Hm @ Sig_m[k]  @ Hm.T
            S   = STT + STm + STm.T + Smm + V

            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue

            # Mahalanobis gating
            if innov @ S_inv @ innov > chi2_gate:
                continue

            # Kalman gains
            P_HT_T = Sig_TT     @ HT.T + Sig_Tm[k] @ Hm.T
            P_HT_m = Sig_Tm[k].T @ HT.T + Sig_m[k] @ Hm.T

            KT = P_HT_T @ S_inv   # 6×4
            Km = P_HT_m @ S_inv   # 3×4

            # Save old covariances
            old_Sig_TT = Sig_TT.copy()
            old_Sig_Tm = Sig_Tm[k].copy()
            old_Sig_m  = Sig_m[k].copy()

            # ----- Landmark update -----
            mu_m[:, k] += Km @ innov
            new_Sig_m   = (I3 - Km @ Hm) @ old_Sig_m - (Km @ HT) @ old_Sig_Tm
            Sig_m[k]    = 0.5 * (new_Sig_m + new_Sig_m.T)

            # ----- Accumulate pose correction -----
            if i >= warmup_steps:
                delta_xi = KT @ innov                    # (6,)

                # Per-feature clip: prevent any single feature causing big jump
                trans_norm = np.linalg.norm(delta_xi[:3])
                rot_norm   = np.linalg.norm(delta_xi[3:])
                if trans_norm > 0.5:                     # max 0.5m per feature
                    delta_xi[:3] *= 0.5 / trans_norm
                if rot_norm > 0.1:                       # max ~6° per feature
                    delta_xi[3:] *= 0.1 / rot_norm

                total_delta_xi += delta_xi
                n_pose_updates += 1

                # Update pose covariance immediately (with clipped delta)
                new_TT = (I6 - KT @ HT) @ old_Sig_TT                          - (KT @ Hm) @ old_Sig_Tm.T
                Sig_TT = 0.5 * (new_TT + new_TT.T)

            # ----- Cross-term update -----
            Sig_Tm[k] = (I6 - KT @ HT) @ old_Sig_Tm                         - (KT @ Hm) @ old_Sig_m

        # Apply averaged pose correction once per timestep → smooth trajectory
        if i >= warmup_steps and n_pose_updates > 0:
            avg_delta_xi = total_delta_xi / n_pose_updates

            # Final clip on the averaged correction
            trans_norm = np.linalg.norm(avg_delta_xi[:3])
            rot_norm   = np.linalg.norm(avg_delta_xi[3:])
            if trans_norm > 0.3:
                avg_delta_xi[:3] *= 0.3 / trans_norm
            if rot_norm > 0.05:
                avg_delta_xi[3:] *= 0.05 / rot_norm

            T = T @ expm(hat(avg_delta_xi))

        T_hist[i] = T

        if i % 200 == 0:
            p = T[:3, 3]
            print(f"  [SLAM] t={t[i]:.1f}s  "
                  f"pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})  "
                  f"lm={init_f.sum()}/{M}")

    return T_hist, mu_m, init_f, sel_idx


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\ucsd\handouts\ece276a\ECE276A_PR3\ECE276A_PR3\data\ECE276A_P3_Data\dataset00"

    dname = os.path.basename(dataset_path)
    print(f"=== Part 4: Visual-Inertial SLAM ===")
    print(f"Dataset: {dataset_path}")

    data = load_dataset(dataset_path)
    t, v, w      = data['t'], data['v'], data['w']
    features     = data['features']
    K_l, K_r     = data['K_l'],   data['K_r']
    iTC_l, iTC_r = data['iTC_l'], data['iTC_r']

    T_hist, mu_m, init_f, sel_idx = vi_slam(
        t, v, w, features, iTC_l, iTC_r, K_l, K_r,
        W_v=5e-5, W_w=1e-5, V_noise=25.0,
        max_features=400, n_update_per_step=15,
        time_subsample_init=3, min_observations=3,
        init_sigma=1.0, innov_gate=50.0,
        chi2_gate=15.87, warmup_steps=50,
    )

    fig = plot_trajectory_and_landmarks(
        T_hist, mu_m, init_f,
        title=f'Part 4 — VI-SLAM ({dname})',
        save_path=f'vi_slam_{dname}.png'
    )
    out_npz = f'vi_slam_result_{dname}.npz'
    np.savez(out_npz, T_hist=T_hist, mu_m=mu_m, init_f=init_f, sel_idx=sel_idx)
    print(f"Saved: {out_npz}")