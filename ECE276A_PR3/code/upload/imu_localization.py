"""
ECE 276A Project 3 — Part 1
IMU Localization via EKF Prediction (SE(3) kinematics)
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from pr3_utils import hat, curlyhat, load_dataset, plot_trajectory_and_landmarks


def imu_ekf_prediction(t, v, w, W_diag=None):
    """
    EKF prediction-only pass using IMU velocity measurements.

    State  : T_t ∈ SE(3) — IMU pose in world frame
    Motion : T_{t+1} = T_t ⊕ exp(τ · [v_t; ω_t]^)
    Cov.   : Σ_{t+1} = F_t Σ_t F_t^T + W

    where   F_t = exp(-τ · curlyhat(u_t))  ∈ R^{6×6}
            W   is the process-noise covariance

    Args:
        t     : (N,)   timestamps [s]
        v     : (3,N)  linear velocity  in body frame
        w     : (3,N)  angular velocity in body frame
        W_diag: (6,)   diagonal of process noise covariance (optional)

    Returns:
        T_hist     : (N, 4, 4)  pose history
        Sigma_hist : (N, 6, 6)  covariance history
    """
    N = len(t)

    # Default process noise
    if W_diag is None:
        # Translational noise slightly larger than rotational
        W_diag = np.array([5e-5, 5e-5, 5e-5, 1e-5, 1e-5, 1e-5])
    W = np.diag(W_diag)

    # Initial state
    T     = np.eye(4)
    Sigma = np.eye(6) * 1e-8

    T_hist     = np.zeros((N, 4, 4))
    Sigma_hist = np.zeros((N, 6, 6))
    T_hist[0]     = T
    Sigma_hist[0] = Sigma

    for i in range(1, N):
        tau = t[i] - t[i - 1]
        u   = np.r_[v[:, i - 1], w[:, i - 1]]   # (6,)

        # ---- mean propagation ----
        T = T @ expm(tau * hat(u))

        # ---- covariance propagation ----
        F = expm(-tau * curlyhat(u))              # 6×6 transition matrix
        Sigma = F @ Sigma @ F.T + W * tau         # scaled by tau for consistency

        T_hist[i]     = T
        Sigma_hist[i] = Sigma

        if i % 500 == 0:
            pos = T[:3, 3]
            print(f"  [IMU] t={t[i]:.1f}s  pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    return T_hist, Sigma_hist


# ============================================================
#  Entry point (standalone run)
# ============================================================

if __name__ == "__main__":
    import sys, os

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\ucsd\handouts\ece276a\ECE276A_PR3\ECE276A_PR3\data\ECE276A_P3_Data\dataset00"

    print(f"=== Part 1: IMU Localization ===")
    print(f"Dataset: {dataset_path}")

    data = load_dataset(dataset_path)
    t, v, w = data['t'], data['v'], data['w']

    T_hist, Sigma_hist = imu_ekf_prediction(t, v, w)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 7))
    xy = T_hist[:, :2, 3]
    ax.plot(xy[:, 0], xy[:, 1], 'b-', linewidth=1.5)
    ax.scatter(xy[0, 0], xy[0, 1], c='g', s=100, zorder=5, label='Start')
    ax.scatter(xy[-1, 0], xy[-1, 1], c='r', s=100, zorder=5, label='End')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Part 1 — IMU-only Trajectory (EKF Prediction)')
    ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    dname = os.path.basename(dataset_path)
    out_path = f"imu_trajectory_{dname}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()

    # Save results
    result_path = f"imu_result_{dname}.npz"
    np.savez(result_path, T_hist=T_hist, Sigma_hist=Sigma_hist, t=t)
    print(f"Results saved: {result_path}")
