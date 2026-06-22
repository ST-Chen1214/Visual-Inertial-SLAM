"""
ECE 276A Project 3 — Main Entry Point
Visual-Inertial SLAM

Usage:
    python main.py [dataset_path] [--part {1,3,4,all}] [--dataset {00,01,02}]

Examples:
    # Run all parts on dataset00
    python main.py --dataset 00

    # Run only Part 1 on a specific path
    python main.py D:/path/to/dataset00 --part 1

    # Run Part 4 VI-SLAM
    python main.py --dataset 01 --part 4
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from pr3_utils import load_dataset, plot_trajectory_and_landmarks, inv_SE3
from imu_localization import imu_ekf_prediction
from landmark_mapping import landmark_mapping_ekf
from vi_slam import vi_slam


# ============================================================
#  Dataset path helper
# ============================================================

BASE_DATA_PATH = r"D:\ucsd\handouts\ece276a\ECE276A_PR3\ECE276A_PR3\data\ECE276A_P3_Data"


def get_dataset_path(dataset_num, base=BASE_DATA_PATH):
    return os.path.join(base, f"dataset{dataset_num:02d}")


# ============================================================
#  Noise presets (tune per dataset)
# ============================================================

NOISE_PRESETS = {
    # V_noise: observation noise variance (pixels^2)
    # Large  -> trust camera less -> stays close to IMU trajectory
    # Small  -> trust camera more -> camera can over-correct IMU

    # dataset00: short ~150m U-path.
    "dataset00": dict(W_v=5e-5, W_w=1e-5, V_noise=50.0),

    # dataset01: ~100m loop.
    "dataset01": dict(W_v=5e-5, W_w=1e-5, V_noise=25.0),

    # dataset02: ~250m path.
    "dataset02": dict(W_v=8e-5, W_w=2e-5, V_noise=25.0),
}

# Per-dataset VI-SLAM extra parameters
# n_update_per_step: more features -> smoother but slower
# chi2_gate: chi-squared threshold for 4 DOF (13.28=99%, 9.49=95%, 7.78=90%)
# innov_gate: max allowed pixel innovation (pre-filter before Mahalanobis)
# warmup_steps: timesteps before camera starts correcting pose
SLAM_PARAMS = {
    "dataset00": dict(max_features=500, n_update_per_step=30,
                      chi2_gate=9.49,  innov_gate=25.0, warmup_steps=80),
    "dataset01": dict(max_features=500, n_update_per_step=30,
                      chi2_gate=11.07, innov_gate=30.0, warmup_steps=50),
    "dataset02": dict(max_features=500, n_update_per_step=30,
                      chi2_gate=11.07, innov_gate=30.0, warmup_steps=50),
}


# ============================================================
#  Run all parts and produce comparison plot
# ============================================================

def run_all(dataset_path, output_dir="."):
    dname = os.path.basename(dataset_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ECE 276A PR3 — {dname}")
    print(f"{'='*60}")

    # ---- Load ----
    data = load_dataset(dataset_path)
    t, v, w   = data['t'], data['v'], data['w']
    features  = data['features']
    K_l, K_r  = data['K_l'],  data['K_r']
    iTC_l, iTC_r = data['iTC_l'], data['iTC_r']

    noise = NOISE_PRESETS.get(dname, NOISE_PRESETS["dataset00"])

    # =============================================
    # Part 1 — IMU EKF Prediction
    # =============================================
    print("\n--- Part 1: IMU Localization ---")
    T_imu, Sig_imu = imu_ekf_prediction(t, v, w,
                                         W_diag=[noise['W_v']]*3 + [noise['W_w']]*3)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    xy = T_imu[:, :2, 3]
    ax1.plot(xy[:, 0], xy[:, 1], 'b-', lw=1.5)
    ax1.scatter(*xy[0], c='g', s=100, zorder=5, label='Start')
    ax1.scatter(*xy[-1], c='r', s=100, zorder=5, label='End')
    ax1.set(title=f'Part 1 — IMU Trajectory ({dname})',
            xlabel='x [m]', ylabel='y [m]')
    ax1.legend(); ax1.axis('equal'); ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    p1_img = os.path.join(output_dir, f"part1_imu_{dname}.png")
    fig1.savefig(p1_img, dpi=150); print(f"  Saved: {p1_img}")
    plt.close(fig1)

    # =============================================
    # Part 3 — Landmark Mapping (fixed trajectory)
    # =============================================
    print("\n--- Part 3: Landmark Mapping ---")
    mu_lm, init_lm, sel_lm = landmark_mapping_ekf(
        T_imu, features, iTC_l, iTC_r, K_l, K_r,
        V_noise=noise['V_noise'],
        max_features=600,
        time_subsample=1,
        min_observations=3,
    )

    fig3 = plot_trajectory_and_landmarks(
        T_imu, mu_lm, init_lm,
        title=f'Part 3 — Landmark Mapping ({dname})',
    )
    p3_img = os.path.join(output_dir, f"part3_landmarks_{dname}.png")
    fig3.savefig(p3_img, dpi=150); print(f"  Saved: {p3_img}")
    plt.close(fig3)

    # =============================================
    # Part 4 — VI-SLAM
    # =============================================
    print("\n--- Part 4: VI-SLAM ---")
    T_slam, mu_slam, init_slam, sel_slam = vi_slam(
        t, v, w, features, iTC_l, iTC_r, K_l, K_r,
        W_v=noise['W_v'],
        W_w=noise['W_w'],
        V_noise=noise['V_noise'],
        max_features=400,
        n_update_per_step=20,
        time_subsample_init=3,
        min_observations=3,
        init_sigma=2.0,
    )

    fig4 = plot_trajectory_and_landmarks(
        T_slam, mu_slam, init_slam,
        title=f'Part 4 — VI-SLAM ({dname})',
    )
    p4_img = os.path.join(output_dir, f"part4_vislam_{dname}.png")
    fig4.savefig(p4_img, dpi=150); print(f"  Saved: {p4_img}")
    plt.close(fig4)

    # =============================================
    # Comparison plot: IMU-only vs VI-SLAM
    # =============================================
    fig_cmp, ax = plt.subplots(figsize=(10, 8))
    xy_imu  = T_imu[:, :2, 3]
    xy_slam = T_slam[:, :2, 3]
    ax.plot(xy_imu[:, 0],  xy_imu[:, 1],  'b--', lw=1.2, label='IMU-only (Part 1)')
    ax.plot(xy_slam[:, 0], xy_slam[:, 1], 'r-',  lw=1.5, label='VI-SLAM (Part 4)')
    if init_slam.sum() > 0:
        lm = mu_slam[:, init_slam]
        ax.scatter(lm[0], lm[1], c='orange', s=2, alpha=0.4, label='Landmarks')
    ax.set(title=f'Trajectory Comparison ({dname})',
           xlabel='x [m]', ylabel='y [m]')
    ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)
    fig_cmp.tight_layout()
    cmp_img = os.path.join(output_dir, f"comparison_{dname}.png")
    fig_cmp.savefig(cmp_img, dpi=150); print(f"  Saved: {cmp_img}")
    plt.close(fig_cmp)

    # ---- Save all results ----
    out_npz = os.path.join(output_dir, f"results_{dname}.npz")
    np.savez(out_npz,
             t=t,
             T_imu=T_imu,
             mu_lm=mu_lm,   init_lm=init_lm,   sel_lm=sel_lm,
             T_slam=T_slam, mu_slam=mu_slam,     init_slam=init_slam)
    print(f"\n  All results saved: {out_npz}")
    print(f"{'='*60}\n")

    return T_imu, T_slam, mu_slam, init_slam


# ============================================================
#  CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='ECE276A PR3 VI-SLAM')
    parser.add_argument('dataset_path', nargs='?', default=None,
                        help='Path to dataset folder (overrides --dataset)')
    parser.add_argument('--dataset', type=str, default='00',
                        choices=['00', '01', '02'],
                        help='Dataset number (00/01/02)')
    parser.add_argument('--part', type=str, default='all',
                        choices=['1', '3', '4', 'all'],
                        help='Which part to run')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for images and .npz files')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = get_dataset_path(int(args.dataset))

    if not os.path.isdir(dataset_path):
        print(f"ERROR: Dataset path not found: {dataset_path}")
        sys.exit(1)

    dname = os.path.basename(dataset_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialise result variables (None until computed)
    T_imu   = None
    mu_lm   = None;  init_lm  = None;  sel_lm  = None
    T_slam  = None;  mu_slam  = None;  init_slam= None;  sel_slam = None

    # Load once
    data = load_dataset(dataset_path)
    t, v, w   = data['t'], data['v'], data['w']
    features  = data['features']
    K_l, K_r  = data['K_l'],  data['K_r']
    iTC_l, iTC_r = data['iTC_l'], data['iTC_r']
    noise = NOISE_PRESETS.get(dname, NOISE_PRESETS["dataset00"])

    if args.part in ('1', 'all'):
        print("\n--- Part 1: IMU Localization ---")
        T_imu, _ = imu_ekf_prediction(t, v, w,
                                       W_diag=[noise['W_v']]*3 + [noise['W_w']]*3)
        np.save(os.path.join(args.output_dir, f"T_imu_{dname}.npy"), T_imu)

        # Plot and save Part 1 trajectory
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        xy = T_imu[:, :2, 3]
        ax1.plot(xy[:, 0], xy[:, 1], 'b-', linewidth=1.5, label='IMU Trajectory')
        ax1.scatter(xy[0, 0], xy[0, 1], c='g', s=100, zorder=5, label='Start')
        ax1.scatter(xy[-1, 0], xy[-1, 1], c='r', s=100, zorder=5, label='End')
        ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
        ax1.set_title(f'Part 1 — IMU Trajectory ({dname})')
        ax1.legend(); ax1.axis('equal'); ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        p1_path = os.path.join(args.output_dir, f"part1_{dname}.png")
        fig1.savefig(p1_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {p1_path}")
        plt.close(fig1)

    if args.part in ('3', 'all'):
        print("\n--- Part 3: Landmark Mapping ---")
        if T_imu is None:
            T_imu, _ = imu_ekf_prediction(t, v, w)
        mu_lm, init_lm, sel_lm = landmark_mapping_ekf(
            T_imu, features, iTC_l, iTC_r, K_l, K_r,
            V_noise=noise['V_noise'], max_features=600,
        )
        fig3 = plot_trajectory_and_landmarks(T_imu, mu_lm, init_lm,
            title=f'Part 3 — Landmark Mapping ({dname})',
            save_path=os.path.join(args.output_dir, f"part3_{dname}.png"))
        plt.close(fig3)

        # Save Part 3 results immediately
        np.save(os.path.join(args.output_dir, f"mu_lm_{dname}.npy"),   mu_lm)
        np.save(os.path.join(args.output_dir, f"init_lm_{dname}.npy"), init_lm)
        print(f"  Saved: mu_lm_{dname}.npy  init_lm_{dname}.npy")

    if args.part in ('4', 'all'):
        print("\n--- Part 4: VI-SLAM ---")
        slam_p = SLAM_PARAMS.get(dname, SLAM_PARAMS["dataset01"])
        T_slam, mu_slam, init_slam, sel_slam = vi_slam(
            t, v, w, features, iTC_l, iTC_r, K_l, K_r,
            **noise, **slam_p,
        )
        fig4 = plot_trajectory_and_landmarks(T_slam, mu_slam, init_slam,
            title=f'Part 4 — VI-SLAM ({dname})',
            save_path=os.path.join(args.output_dir, f"part4_{dname}.png"))
        plt.close(fig4)

        # Save Part 4 results immediately
        np.save(os.path.join(args.output_dir, f"T_slam_{dname}.npy"),    T_slam)
        np.save(os.path.join(args.output_dir, f"mu_slam_{dname}.npy"),   mu_slam)
        np.save(os.path.join(args.output_dir, f"init_slam_{dname}.npy"), init_slam)
        print(f"  Saved: T_slam_{dname}.npy  mu_slam_{dname}.npy  init_slam_{dname}.npy")

    # ---- Save combined results_*.npz for trajectory_video.py ----
    npz_path = os.path.join(args.output_dir, f"results_{dname}.npz")
    save_dict = {'t': t}
    if T_imu is not None:
        save_dict['T_imu'] = T_imu
    if mu_lm is not None:
        save_dict['mu_lm']   = mu_lm
        save_dict['init_lm'] = init_lm
        save_dict['sel_lm']  = sel_lm
    if T_slam is not None:
        save_dict['T_slam']    = T_slam
        save_dict['mu_slam']   = mu_slam
        save_dict['init_slam'] = init_slam
        save_dict['sel_slam']  = sel_slam
    np.savez(npz_path, **save_dict)
    print(f"  Results saved: {npz_path}")
    print(f"  Keys: {list(save_dict.keys())}")

    print("\nDone!")