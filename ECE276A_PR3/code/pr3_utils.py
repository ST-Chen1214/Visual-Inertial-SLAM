"""
ECE 276A Project 3 - Utility Functions
Visual-Inertial SLAM helper math and data loading tools
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import os


# ============================================================
#  SE(3) / Lie Algebra Utilities
# ============================================================

def skew(v):
    """Convert 3-vector to 3×3 skew-symmetric matrix."""
    v = v.flatten()
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])


def hat(u):
    """
    Convert 6-vector u = [rho; phi] to 4×4 se(3) twist matrix.
      hat(u) = [[skew(phi), rho],
                [0^T,       0  ]]
    """
    rho = u[:3].flatten()
    phi = u[3:].flatten()
    return np.block([[skew(phi), rho.reshape(3, 1)],
                     [np.zeros((1, 3)), 0.0        ]])


def curlyhat(u):
    """
    6×6 'adjoint' (curly hat / ad) of a twist u = [rho; phi].
    Used for covariance propagation in SE(3).
      curlyhat(u) = [[skew(phi), skew(rho)],
                     [0,         skew(phi)]]
    """
    rho = u[:3].flatten()
    phi = u[3:].flatten()
    return np.block([[skew(phi), skew(rho)],
                     [np.zeros((3, 3)), skew(phi)]])


def adjoint_SE3(T):
    """
    6×6 Adjoint of T ∈ SE(3).
    Ad_T = [[R, p× R],
            [0,  R  ]]
    """
    R = T[:3, :3]
    p = T[:3, 3]
    return np.block([[R, skew(p) @ R],
                     [np.zeros((3, 3)), R]])


def inv_SE3(T):
    """Efficient inversion of SE(3) matrix (avoids full matrix inverse)."""
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ p
    return Ti


def circle_dot(q):
    """
    4×6 matrix such that d/dxi [exp(hat(xi)) @ q]|_{xi=0} = circle_dot(q).
    q is a 4-vector (homogeneous point or direction).
    """
    mat = np.zeros((4, 6))
    mat[:3, :3] = np.eye(3)
    mat[:3, 3:] = -skew(q[:3])
    return mat


# ============================================================
#  Stereo Camera Utilities
# ============================================================

# Global flag: set by detect_optical_axis() after loading data
# True  = camera x-axis is optical axis (this dataset: extL_T_imu R=I)
# False = camera z-axis is optical axis (standard ROS/OpenCV convention)
_X_IS_OPTICAL_AXIS = False

def detect_optical_axis(iTC_l):
    """
    Detect whether camera x or z is the optical axis.
    Call once after loading extrinsics.
    Sets global _X_IS_OPTICAL_AXIS.
    
    Rule: if camera z-axis (in IMU frame) does NOT point forward (IMU-x),
          then x is the optical axis.
    """
    global _X_IS_OPTICAL_AXIS
    R_cam_T_imu = inv_SE3(iTC_l)[:3, :3]   # = extL[:3,:3]
    # cam z-axis expressed in IMU frame
    cam_z_in_imu = R_cam_T_imu.T @ np.array([0, 0, 1])
    # If cam-z is NOT forward (imu-x), use x as optical axis
    _X_IS_OPTICAL_AXIS = abs(cam_z_in_imu[0]) < 0.5
    axis_name = "x" if _X_IS_OPTICAL_AXIS else "z"
    print(f"[detect_optical_axis] cam_z_in_imu={cam_z_in_imu}  "
          f"=> optical axis = {axis_name}")
    return _X_IS_OPTICAL_AXIS


def _pinhole(p, K):
    """
    Pinhole projection. Uses global _X_IS_OPTICAL_AXIS to pick convention.
      x-axis optical: proj = K @ [y/x, z/x, 1]
      z-axis optical: proj = K @ [x/z, y/z, 1]
    """
    if _X_IS_OPTICAL_AXIS:
        if p[0] <= 0:
            return None
        uv = K @ np.array([p[1]/p[0], p[2]/p[0], 1.0])
    else:
        if p[2] <= 0:
            return None
        uv = K @ np.array([p[0]/p[2], p[1]/p[2], 1.0])
    return uv[:2]


def _optical_depth(p):
    """Return depth along optical axis."""
    return p[0] if _X_IS_OPTICAL_AXIS else p[2]


def project_stereo(m_w, T_imu, iTC_l, iTC_r, K_l, K_r):
    """
    Project world-frame landmark m_w through stereo cameras.
    Supports datasets where camera x-axis is the optical axis.

    Returns:
        z = [lx, ly, rx, ry]  or  None if behind camera
    """
    Ti    = inv_SE3(T_imu)
    cTi_l = inv_SE3(iTC_l)
    cTi_r = inv_SE3(iTC_r)

    m_h = np.r_[m_w, 1.0]
    p_l = cTi_l @ Ti @ m_h
    p_r = cTi_r @ Ti @ m_h

    zl = _pinhole(p_l[:3], K_l)
    zr = _pinhole(p_r[:3], K_r)
    if zl is None or zr is None:
        return None
    return np.r_[zl, zr]


def _Dpi(p, K):
    """
    2×3 Jacobian of pinhole projection d(proj)/d(p).
    Uses global _X_IS_OPTICAL_AXIS.
    """
    x, y, z = p[0], p[1], p[2]
    fu, fv = K[0,0], K[1,1]
    if _X_IS_OPTICAL_AXIS:
        # x is optical axis: proj = [fu*y/x + cu, fv*z/x + cv]
        return np.array([[-fu*y/x**2, fu/x, 0    ],
                         [-fv*z/x**2, 0,    fv/x ]])
    else:
        # z is optical axis: proj = [fu*x/z + cu, fv*y/z + cv]
        return np.array([[fu/z, 0,    -fu*x/z**2],
                         [0,    fv/z, -fv*y/z**2]])


def jacobian_landmark(m_w, T_imu, iTC_l, iTC_r, K_l, K_r):
    """
    Jacobian of stereo observation z w.r.t. landmark position m_w.
    Returns H of shape (4, 3).
    """
    Ti    = inv_SE3(T_imu)
    cTi_l = inv_SE3(iTC_l)
    cTi_r = inv_SE3(iTC_r)
    m_h   = np.r_[m_w, 1.0]

    cTw_l = cTi_l @ Ti
    cTw_r = cTi_r @ Ti
    p_l   = (cTw_l @ m_h)[:3]
    p_r   = (cTw_r @ m_h)[:3]

    Hl = _Dpi(p_l, K_l) @ cTw_l[:3, :3]   # 2×3
    Hr = _Dpi(p_r, K_r) @ cTw_r[:3, :3]   # 2×3
    return np.vstack([Hl, Hr])              # 4×3


def jacobian_pose(m_w, T_imu, iTC_l, iTC_r, K_l, K_r):
    """
    Jacobian of stereo observation z w.r.t. IMU pose perturbation xi.
    Perturbation model: T\'  = T @ exp(hat(xi))
    Returns H of shape (4, 6).
    """
    Ti    = inv_SE3(T_imu)
    cTi_l = inv_SE3(iTC_l)
    cTi_r = inv_SE3(iTC_r)
    m_h   = np.r_[m_w, 1.0]

    q   = Ti @ m_h                     # point in IMU frame (4,)
    dq_dxi = -circle_dot(q)            # 4×6

    cTw_l = cTi_l @ Ti
    cTw_r = cTi_r @ Ti
    p_l   = (cTw_l @ m_h)[:3]
    p_r   = (cTw_r @ m_h)[:3]

    dp_l = cTi_l[:3, :] @ dq_dxi      # 3×6
    dp_r = cTi_r[:3, :] @ dq_dxi      # 3×6

    Hl = _Dpi(p_l, K_l) @ dp_l        # 2×6
    Hr = _Dpi(p_r, K_r) @ dp_r        # 2×6
    return np.vstack([Hl, Hr])         # 4×6


# ============================================================
#  Landmark Triangulation (DLT)
# ============================================================

def triangulate(z_obs, T_imu, iTC_l, iTC_r, K_l, K_r,
                min_depth=0.3, max_depth=150.0, max_reproj_err=10.0):
    """
    Triangulate world-frame position of a landmark from a stereo observation.
    Uses the Direct Linear Transform (DLT / SVD) with strict quality filtering.

    Returns:
        m: (3,) world-frame position, or None if degenerate / out of range.
    """
    Ti    = inv_SE3(T_imu)
    cTw_l = inv_SE3(iTC_l) @ Ti
    cTw_r = inv_SE3(iTC_r) @ Ti

    lx, ly, rx, ry = z_obs

    # Disparity sanity check: right x should be less than left x
    disparity = lx - rx
    if disparity <= 0.1:          # negative or zero disparity
        return None
    if disparity > 300:           # unrealistically large
        return None

    # DLT triangulation — supports both optical axis conventions
    if _X_IS_OPTICAL_AXIS:
        # x is optical axis: u = fu*(py/px)+cu, v = fv*(pz/px)+cv
        # DLT rows: (u-cu)*row0 - fu*row1 = 0
        #           (v-cv)*row0 - fv*row2 = 0
        fu_l, fv_l = K_l[0,0], K_l[1,1]
        cu_l, cv_l = K_l[0,2], K_l[1,2]
        fu_r, fv_r = K_r[0,0], K_r[1,1]
        cu_r, cv_r = K_r[0,2], K_r[1,2]
        A = np.array([
            (lx-cu_l)*cTw_l[0,:] - fu_l*cTw_l[1,:],
            (ly-cv_l)*cTw_l[0,:] - fv_l*cTw_l[2,:],
            (rx-cu_r)*cTw_r[0,:] - fu_r*cTw_r[1,:],
            (ry-cv_r)*cTw_r[0,:] - fv_r*cTw_r[2,:],
        ])
    else:
        # z is optical axis: standard DLT
        P_l = K_l @ cTw_l[:3, :]
        P_r = K_r @ cTw_r[:3, :]
        A = np.array([lx * P_l[2] - P_l[0],
                      ly * P_l[2] - P_l[1],
                      rx * P_r[2] - P_r[0],
                      ry * P_r[2] - P_r[1]])

    _, _, Vt = np.linalg.svd(A)
    ph = Vt[-1]
    if abs(ph[3]) < 1e-10:
        return None
    ph /= ph[3]
    m = ph[:3]
    if not np.all(np.isfinite(m)):
        return None

    # ---- depth check in left/right camera (auto-detect optical axis) ----
    p_l = (inv_SE3(iTC_l) @ Ti @ ph)
    p_r = (inv_SE3(iTC_r) @ Ti @ ph)
    depth_l = _optical_depth(p_l[:3])
    depth_r = _optical_depth(p_r[:3])
    if depth_l < min_depth or depth_l > max_depth:
        return None
    if depth_r < min_depth or depth_r > max_depth:
        return None

    # ---- reprojection error check ----
    zl = _pinhole(p_l[:3], K_l)
    zr = _pinhole(p_r[:3], K_r)
    if zl is None or zr is None:
        return None
    err_l = np.linalg.norm(zl - np.array([lx, ly]))
    err_r = np.linalg.norm(zr - np.array([rx, ry]))
    if err_l > max_reproj_err or err_r > max_reproj_err:
        return None

    return m




# ============================================================
#  Extrinsic Auto-Detection Helper
# ============================================================

def _reproj_error_batch(features, T_imu, iTC_l, iTC_r, K_l, K_r, n_test=30):
    """
    Compute mean reprojection error over n_test valid features at t=0.
    Used to auto-detect correct extrinsic direction.
    """
    Ti    = inv_SE3(T_imu)
    cTw_l = inv_SE3(iTC_l) @ Ti
    cTw_r = inv_SE3(iTC_r) @ Ti

    # features shape: (4, M, N)
    z0 = features[:, :, 0]
    valid = np.where(~np.any(z0 == -1, axis=0))[0]
    if len(valid) == 0:
        return 1e9

    rng = np.random.default_rng(0)
    sample = rng.choice(valid, min(n_test, len(valid)), replace=False)

    errors = []
    for k in sample:
        lx, ly, rx, ry = z0[:, k]
        # DLT triangulation
        P_l = K_l @ cTw_l[:3, :]
        P_r = K_r @ cTw_r[:3, :]
        A = np.array([lx * P_l[2] - P_l[0],
                      ly * P_l[2] - P_l[1],
                      rx * P_r[2] - P_r[0],
                      ry * P_r[2] - P_r[1]])
        _, _, Vt = np.linalg.svd(A)
        ph = Vt[-1]
        if abs(ph[3]) < 1e-10:
            continue
        ph /= ph[3]
        p_l = cTw_l @ ph
        p_r = cTw_r @ ph
        if p_l[2] <= 0 or p_r[2] <= 0:
            errors.append(1e6)
            continue
        proj_l = K_l @ (p_l[:3] / p_l[2])
        proj_r = K_r @ (p_r[:3] / p_r[2])
        e = (np.linalg.norm(proj_l[:2] - [lx, ly]) +
             np.linalg.norm(proj_r[:2] - [rx, ry])) / 2
        errors.append(e)

    return float(np.mean(errors)) if errors else 1e9


# ============================================================
#  Data Loading
# ============================================================

def load_dataset(dataset_path):
    """
    Load a PR3 dataset .npy file.

    Handles multiple possible key naming conventions.
    Returns a dict with standardised keys:
        t        : (N,)    timestamps
        v        : (3,N)   linear  velocity in IMU body frame
        w        : (3,N)   angular velocity in IMU body frame
        features : (4,M,N) stereo pixel coords; -1 = not observed
        K_l      : (3,3)   left  camera intrinsics
        K_r      : (3,3)   right camera intrinsics
        iTC_l    : (4,4)   IMU <- left  camera (extrinsic)
        iTC_r    : (4,4)   IMU <- right camera (extrinsic)
    """
    # Find .npy file
    npy_files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')
                 and 'img' not in f]
    if not npy_files:
        raise FileNotFoundError(f"No .npy file found in {dataset_path}")

    npy_path = os.path.join(dataset_path, npy_files[0])
    print(f"[load] Reading {npy_path}")
    raw = np.load(npy_path, allow_pickle=True).item()
    print(f"[load] Keys found: {list(raw.keys())}")

    def pick(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f"None of {keys} found in data. Available: {list(d.keys())}")

    t  = pick(raw, 'timestamps', 't', 't_stamps').flatten()
    v  = pick(raw, 'v_t', 'v', 'linear_velocity', 'lin_vel')
    w  = pick(raw, 'w_t', 'w', 'angular_velocity', 'ang_vel')
    ft = pick(raw, 'features', 'z', 'z_obs')

    K_l   = pick(raw, 'K_l', 'K0', 'cam0_intrinsics')
    K_r   = pick(raw, 'K_r', 'K1', 'cam1_intrinsics')

    # ---- Extrinsic: auto-detect correct direction via reprojection test ----
    raw_keys = list(raw.keys())
    if 'extL_T_imu' in raw_keys:
        extL_raw = raw['extL_T_imu']
        extR_raw = raw['extR_T_imu']
        # Candidate A: extL_T_imu = cam_T_imu  -> iTC = inv(extL)
        cand_A_l = inv_SE3(extL_raw)
        cand_A_r = inv_SE3(extR_raw)
        # Candidate B: extL_T_imu = imu_T_cam  -> iTC = extL directly
        cand_B_l = extL_raw
        cand_B_r = extR_raw
        # Auto-select by reprojection error
        err_A = _reproj_error_batch(ft, np.eye(4), cand_A_l, cand_A_r, K_l, K_r)
        err_B = _reproj_error_batch(ft, np.eye(4), cand_B_l, cand_B_r, K_l, K_r)
        print(f"[load] Extrinsic test -> A(inv) err={err_A:.2f}px  B(raw) err={err_B:.2f}px")
        if err_A <= err_B:
            iTC_l, iTC_r = cand_A_l, cand_A_r
            print("[load] Using iTC = inv(extL_T_imu)")
        else:
            iTC_l, iTC_r = cand_B_l, cand_B_r
            print("[load] Using iTC = extL_T_imu (as-is)")
    else:
        iTC_l = pick(raw, 'imu_T_cam0', 'imu_T_cam_l', 'iTC_l', 'imuTcam0')
        iTC_r = pick(raw, 'imu_T_cam1', 'imu_T_cam_r', 'iTC_r', 'imuTcam1')

    # Normalise shapes
    if v.shape[0] != 3:
        v = v.T
    if w.shape[0] != 3:
        w = w.T

    # features should be (4, M, N)
    if ft.ndim == 3 and ft.shape[0] != 4:
        ft = ft.transpose(2, 1, 0)   # (N,M,4) -> (4,M,N)

    # Auto-detect camera optical axis from extrinsics
    detect_optical_axis(iTC_l)
    print(f"[load] N={t.shape[0]}  M={ft.shape[1]}  duration={t[-1]-t[0]:.1f}s")
    return dict(t=t, v=v, w=w, features=ft,
                K_l=K_l, K_r=K_r, iTC_l=iTC_l, iTC_r=iTC_r)


# ============================================================
#  Visualisation
# ============================================================

def plot_trajectory_and_landmarks(T_hist, mu_m, init,
                                   title="VI-SLAM Result", save_path=None,
                                   lm_margin=30.0):
    """
    Plot the estimated IMU trajectory (x-y plane) and landmark positions.

    Args:
        lm_margin : landmarks further than this many metres from the
                    trajectory bounding box are hidden for display only.
                    No data is modified.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Trajectory
    xy = T_hist[:, :2, 3]
    ax.plot(xy[:, 0], xy[:, 1], 'b-', linewidth=1.5, label='IMU Trajectory')
    ax.scatter(xy[0, 0], xy[0, 1], c='g', s=80, zorder=5, label='Start')
    ax.scatter(xy[-1, 0], xy[-1, 1], c='r', s=80, zorder=5, label='End')

    # Landmarks - filter outliers for display only
    if init.sum() > 0:
        lm = mu_m[:2, init]          # (2, M_init) x-y only

        # Trajectory bounding box + margin
        x_min = xy[:, 0].min() - lm_margin
        x_max = xy[:, 0].max() + lm_margin
        y_min = xy[:, 1].min() - lm_margin
        y_max = xy[:, 1].max() + lm_margin

        in_box = ((lm[0] >= x_min) & (lm[0] <= x_max) &
                  (lm[1] >= y_min) & (lm[1] <= y_max))

        lm_show  = lm[:, in_box]
        n_total  = int(init.sum())
        n_show   = int(in_box.sum())

        ax.scatter(lm_show[0], lm_show[1],
                   c='orange', s=3, alpha=0.6,
                   label=f'Landmarks ({n_show}/{n_total})')

        if n_show < n_total:
            print(f"[plot] Showing {n_show}/{n_total} landmarks "
                  f"(within +/-{lm_margin:.0f}m of trajectory)")

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[plot] Saved to {save_path}")
    plt.show()
    return fig
